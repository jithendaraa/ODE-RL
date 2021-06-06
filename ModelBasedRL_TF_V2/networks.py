import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import tools
from classification_models.tfkeras import Classifiers

class RSSM(tools.Module):

  def __init__(
      self, stoch=32, spatial=8, hidden=32, kernel=5, layers_input=1, layers_output=1,
      rec_depth=1, shared=False, discrete=False, act=tf.nn.elu, skip=True,
      mean_act='none', std_act='softplus', temp_post=True, min_std=0.1,
      cell='keras'):
    super().__init__()
    self._min_std = min_std
    self._layers_input = layers_input
    self._layers_output = layers_output
    self._rec_depth = rec_depth
    self._shared = shared
    self._discrete = discrete
    self._act = act
    self._mean_act = mean_act
    self._std_act = std_act
    self._temp_post = temp_post
    self._embed = None
    self._spatial_size = spatial
    self._stoch_channels = stoch
    self._hidden_channels = hidden
    self._kernel = kernel
    self._skip = skip

    if cell == 'convgru':
      self._cell = ConvGRUCell(spatial=self._spatial_size, depth=self._hidden_channels, kernel=self._kernel)
      self._is_cell_stochastic = False
    elif cell == 's_convgru':
      self._cell = StochasticConvGRUCell(spatial=self._spatial_size, depth=self._hidden_channels, kernel=self._kernel, skip=self._skip)
      self._is_cell_stochastic = True
    else:
      raise NotImplementedError(cell)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._spatial_size, self._spatial_size, self._stoch_channels], dtype),
          stoch=tf.zeros([batch_size, self._spatial_size, self._spatial_size, self._stoch_channels], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._spatial_size, self._spatial_size, self._stoch_channels], dtype),
          std=tf.zeros([batch_size, self._spatial_size, self._spatial_size, self._stoch_channels], dtype),
          stoch=tf.zeros([batch_size, self._spatial_size, self._spatial_size, self._stoch_channels], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))

    if self._is_cell_stochastic:
      state['u_sample'] = tf.zeros([batch_size, self._hidden_channels], dtype)
      state['u_prob'] = tf.zeros([batch_size, self._hidden_channels], dtype)
      state['u_logit'] = tf.zeros([batch_size, self._hidden_channels], dtype)

    return state

  @tf.function
  def observe(self, embed, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(tf.shape(embed)[0])
    embed = swap(embed)
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs),
        (embed, embed), (state, state))
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  @tf.function
  def imagine(self, batch_size, state=None):
    swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(batch_size[0])
    temp = self.initial(batch_size[1])
    assert isinstance(state, dict), state
    prior = tools.static_scan(self.img_step, temp, state)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_feat(self, state, reshape=False):
    stoch = state['stoch']
    # if self._discrete:
    #   shape = stoch.shape[:-2] + [self._stoch * self._discrete]
    #   stoch = tf.reshape(stoch, shape)
    out = tf.concat([stoch, state['deter']], -1)
    if reshape:
      return tools._convert_dense(out)
    return out

  def get_dist(self, state, dtype=None):
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      n_dims = len(logit.shape)
      logit = tf.transpose(logit, perm=list(range(len(logit.shape[:-3]))) + [n_dims-1, n_dims-3, n_dims-2])
      logit = tf.reshape(logit, list(logit.shape[:-2]) + [self._spatial_size * self._spatial_size])
      dist = tfd.Independent(tools.OneHotDist(logit), 1)
      if dtype != tf.float32:
        dist = tools.DtypeDist(dist, dtype or state['logit'].dtype)

    else:
      mean, std = state['mean'], state['std']
      n_dims = len(mean.shape)
      mean = tf.transpose(mean, perm=list(range(len(mean.shape[:-3]))) + [n_dims-1, n_dims-3, n_dims-2])
      mean = tf.reshape(mean, list(mean.shape[:-2]) + [self._spatial_size * self._spatial_size])

      std = tf.transpose(std, perm=list(range(len(std.shape[:-3]))) + [n_dims-1, n_dims-3, n_dims-2])
      std = tf.reshape(std, list(std.shape[:-2]) + [self._spatial_size * self._spatial_size])
      if dtype:
        mean = tf.cast(mean, dtype)
        std = tf.cast(std, dtype)
      dist = tools.DtypeDist(tfd.MultivariateNormalDiag(mean, std))
    return dist

  def dist_sample(self, state, sample=None, dtype=None):
    dist = self.get_dist(state, dtype)
    if sample:
      out = dist.sample()
    else:
      out = dist.mode()
    n_dims = len(out.shape) + 1
    out = tf.reshape(out, list(out.shape[:-1]) + [self._spatial_size, self._spatial_size])
    out = tf.transpose(out, list(range(n_dims-3)) + [n_dims-2, n_dims-1, n_dims-3])
    return out

  @tf.function
  def obs_step(self, prev_state, temp, embed, sample=True):
    if not self._embed:
      self._embed = embed.shape[-1]

    prior = self.img_step(prev_state, None, None, sample)
    if self._shared:
      post = self.img_step(prev_state, None, embed, sample)
    else:
      if self._temp_post:
        x = tf.concat([prior['deter'], embed], -1)
      else:
        x = embed
      for i in range(self._layers_output):
        x = self.get(f'obi{i}', tfkl.Conv2D, self._hidden_channels, self._kernel, padding='same', activation=self._act)(x)
        #x = self.get(f'obi{i}', tfkl.Dense, self._hidden, self._act)(x)
      # if self._attention is not None:
      #   x = self._attention_post(x)

      stats = self._suff_stats_layer('obs', x)
      # if sample:
      stoch = self.dist_sample(stats, sample)
      # else:
      #   stoch = self.dist_sample(stats, sample)
      post = {'stoch': stoch, 'deter': prior['deter'], **stats}

    if self._is_cell_stochastic:
        post['u_sample'] = prior['u_sample']
        post['u_logit']  = prior['u_logit']
        post['u_prob']   = prior['u_prob']

    return post, prior

  @tf.function
  def img_step(self, prev_state, temp, embed=None, sample=True):
    prev_stoch = prev_state['stoch']

    # if self._discrete:
    #   shape = prev_stoch.shape[:-2] + [self._stoch * self._discrete]
    #   prev_stoch = tf.reshape(prev_stoch, shape)

    if self._shared:
      if embed is None:
        shape = prev_stoch.shape[:-1] + [self._embed]
        embed = tf.zeros(shape, prev_stoch.dtype)
      x = tf.concat([prev_stoch, embed], -1)
    else:
      x = prev_stoch
    for i in range(self._layers_input):
      x = self.get(f'ini{i}', tfkl.Conv2D, self._hidden_channels, self._kernel, padding='same', activation=self._act)(x)
    for _ in range(self._rec_depth):
      if self._is_cell_stochastic:
        deter = [prev_state['deter'], prev_state['u_sample'], prev_state['u_prob'], prev_state['u_logit']]
      else:
        deter = [prev_state['deter']]
      # print ('before cell ', prev_state['u_sample'].shape, prev_state['u_logit'].shape)
      x, deter = self._cell(x, deter)
      # deter = deter[0]  # Keras wraps the state in a list.

    for i in range(self._layers_output):
      x = self.get(f'imo{i}', tfkl.Conv2D, self._hidden_channels, self._kernel, padding='same', activation=self._act)(x)

    # if self._attention is not None:
    #     x = self._attention_prior(x)

    stats = self._suff_stats_layer('ims', x)
    stoch = self.dist_sample(stats, sample)

    prior = {'stoch': stoch, 'deter': deter[0], **stats}

    if self._is_cell_stochastic:
      prior['u_sample'] = deter[1]
      prior['u_prob'] = deter[2]
      prior['u_logit'] = deter[3]

    return prior

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Conv2D, self._stoch_channels, self._kernel, padding='same', activation=None)(x)
      # logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': x}
    else:
      # print ('here, latent is gaussian ')
      x = self.get(name, tfkl.Conv2D, 2 * self._stoch_channels, self._kernel, padding='same', activation=None)(x)
      # x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      mean = {
          'none': lambda: mean,
          'tanh5': lambda: 5.0 * tf.math.tanh(mean / 5.0),
      }[self._mean_act]()
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'abs': lambda: tf.math.abs(std + 1),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def sparsity_loss(self, post, prior_prob, free, scale, forward=True):
    kld = tfd.kl_divergence
    u_post  = tfd.Independent(tools.BernoulliDist(probs=tf.cast(post['u_logit'], tf.float32)), 1)
    u_prior = tfd.Independent(tools.BernoulliDist(probs=tf.ones(post['u_logit'].shape) * prior_prob), 1)
    if forward:
      loss = kld(u_prior, u_post)
    else:
      loss = kld(u_post, u_prior)

    loss = tf.maximum(tf.reduce_mean(loss), free)
    return loss * scale

  def kl_loss(self, post, prior, forward, balance, free, scale):
    kld = tfd.kl_divergence
    if self._discrete:
      dist = lambda x: self.get_dist(x, tf.float32)
    else:
      dist = lambda x: self.get_dist(x, tf.float32)._dist
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(dist(lhs), dist(rhs))
      loss = tf.reduce_mean(tf.maximum(value, free))
    else:
      value_lhs = value = kld(dist(lhs), dist(sg(rhs)))
      value_rhs = kld(dist(sg(lhs)), dist(rhs))
      loss_lhs = tf.maximum(tf.reduce_mean(value_lhs), free)
      loss_rhs = tf.maximum(tf.reduce_mean(value_rhs), free)
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    loss *= scale
    return loss, value


class ConvEncoder(tools.Module):

  def __init__(
      self, depth=32, act=tf.nn.relu, kernels=(4, 4, 4, 4)):
    self._act = act
    self._depth = depth
    self._kernels = kernels

  def __call__(self, obs):
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** i * self._depth
      x = self._act(self.get(f'h{i}', tfkl.Conv2D, depth, kernel, 2)(x))
    x = tf.reshape(x, [x.shape[0], np.prod(x.shape[1:])])
    shape = tf.concat([tf.shape(obs['image'])[:-3], [x.shape[-1]]], 0)
    # print('Encoder output:', shape)
    return tf.reshape(x, shape)

class ResNetPretrained(tools.Module):

  def __init__(self, version='resnet50', img_size=(224, 224), out_dim=32, shape=(8, 8)):
    if version == 'resnet50':
      self._resnet_encoder = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(img_size) + (3, ))
    elif version in ['resnet18', 'resnet34']:
      ResNet, _ = Classifiers.get(version)
      self._resnet_encoder = ResNet(include_top=False, weights='imagenet', input_shape=(img_size) + (3, ))
    else:
      raise NotImplementedError(version)
    self._resnet_encoder.trainable = False
    self._img_size = img_size
    self._act = tf.nn.relu
    self._out_dim = out_dim
    self._shape = shape
    print(f"ResNet img_size: {img_size}, shape: {shape}")

  def __call__(self, obs):
    x = tf.reshape(obs['resnet_image'], (-1,) + tuple(obs['resnet_image'].shape[-3:]))
    x = self._resnet_encoder(x)
    if self._img_size[0] == 224 and self._shape[0] == 7:
      x = self._act(self.get('resnet_conv', tfkl.Conv2D, self._out_dim, 3, padding='same')(x))
    else:
      x = tools._convert_dense(x)
      x = self._act(self.get('resnet_linear', tfkl.Dense, np.prod(self._shape) * self._out_dim, None)(x))
    shape = obs['resnet_image'].shape[:-3] + self._shape + (self._out_dim, )
    return tf.reshape(x, shape)


class ImpalaCNN(tools.Module):

  def __init__(self, depths=[16, 32, 32], flatten_attn=None, attention=None, spatial=6):
    self._depths = depths
    self._act = tf.nn.relu
    self._flatten_attn = flatten_attn
    if self._flatten_attn == 'c':
      self._flatten_attn_model = FlattenAttnC(8, 32)
    elif self._flatten_attn == 's':
      self._flatten_attn_model = FlattenAttnS(8, 32)

    if attention == 'slot':
      self._attention_layer = SlotAttention(spatial)
    else:
      self._attention_layer = None

  def residual_block(self, obs, i, j):
    depth = obs.shape[-1]
    out = self._act(obs)
    out = self._act(self.get(f'impala_res{i}_{j}_conv1', tfkl.Conv2D, depth, 3, padding='same')(out))
    out = self.get(f'impala_res{i}_{j}_conv2', tfkl.Conv2D, depth, 3, padding='same')(out)
    return out + obs

  def __call__(self, obs):
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    for i, depth in enumerate(self._depths):
      x = self.get(f'impala_conv{i}', tfkl.Conv2D, depth, 3, padding='same')(x)
      x = self.get(f'impala_max_pool{i}', tfkl.MaxPool2D, pool_size=3, strides=2, padding='same')(x)
      x = self.residual_block(x, i, 0)
      x = self.residual_block(x, i, 1)
    x = self._act(x)
    if self._attention_layer:
      # print ('added attention here')
      x = self._attention_layer(x)

    # x = self._act(self.get(f'impala_new_layer{i}', tfkl.Conv2D, self._depths[-1], 3)(x))

    if self._flatten_attn == 'c' or self._flatten_attn == 's':
      x = self._flatten_attn_model(x)

    shape = obs['image'].shape[:-3] + x.shape[-3:]
    # print('Encoder output:', shape)
    return tf.reshape(x, shape)

class FlattenAttnC(tools.Module):

  def __init__(self, spatial, channels):
    self._spatial = spatial
    self._channels = channels
    self._act = tf.nn.relu

  def __call__(self, x):
    out = []
    old_shape = x.shape[:-1]
    for i in range(self._channels):
      temp = self.get(f'flatten_l{i}', tfkl.Flatten)(x[..., i])
      temp = self.get(f'flatten_attn{i}', tfkl.Dense, self._spatial * self._spatial, activation=self._act)(temp)
      temp = tf.reshape(temp, old_shape)
      out.append(temp)
    return tf.stack(out, -1)

class FlattenAttnS(tools.Module):

  def __init__(self, spatial, channels):
    self._spatial = spatial
    self._channels = channels
    self._act = tf.nn.relu

  def __call__(self, x):
    out = []
    old_shape = x.shape[:-1]
    for i in range(self._channels):
      temp = self.get(f'flatten_l', tfkl.Flatten)(x[..., i])
      temp = self.get(f'flatten_attn', tfkl.Dense, self._spatial * self._spatial, activation=self._act)(temp)
      temp = tf.reshape(temp, old_shape)
      out.append(temp)

    return tf.stack(out, -1)

class ConvDecoder(tools.Module):

  def __init__(
      self, depth=32, act=tf.nn.relu, shape=(64, 64, 3), kernels=(5, 5, 6, 6),
      thin=True, resnet=False):
    self._act = act
    self._depth = depth
    self._shape = shape
    self._kernels = kernels
    self._thin = thin
    self._resnet = resnet

  def __call__(self, features, dtype=None):
    ConvT = tfkl.Conv2DTranspose
    # features = tools._convert_dense(features)
    if self._resnet:
      x = self.get('hin', tfkl.Dense, 11 * 11 * 32, None)(features)
      x = tf.reshape(x, [-1, 11, 11, 32])
    else:
      if self._thin:
        x = self.get('hin', tfkl.Dense, 32 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
      else:
        x = self.get('hin', tfkl.Dense, 128 * self._depth, None)(features)
        x = tf.reshape(x, [-1, 2, 2, 32 * self._depth])
    for i, kernel in enumerate(self._kernels):
      depth = 2 ** (len(self._kernels) - i - 2) * self._depth
      act = self._act
      if i == len(self._kernels) - 1:
        depth = self._shape[-1]
        act = None
      x = self.get(f'h{i}', ConvT, depth, kernel, 2, activation=act)(x)
    # print('Decoder output:', x.shape)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    if dtype:
      mean = tf.cast(mean, dtype)
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))

class DenseHead(tools.Module):

  def __init__(
      self, shape, layers, units, act=tf.nn.elu, dist='normal', std=1.0):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._act = act
    self._dist = dist
    self._std = std

  def __call__(self, features, dtype=None):
    x = features
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    mean = self.get(f'hmean', tfkl.Dense, np.prod(self._shape))(x)
    mean = tf.reshape(mean, tf.concat(
        [tf.shape(features)[:-1], self._shape], 0))
    if self._std == 'learned':
      std = self.get(f'hstd', tfkl.Dense, np.prod(self._shape))(x)
      std = tf.nn.softplus(std) + 0.01
      std = tf.reshape(std, tf.concat(
          [tf.shape(features)[:-1], self._shape], 0))
    else:
      std = self._std
    if dtype:
      mean, std = tf.cast(mean, dtype), tf.cast(std, dtype)
    if self._dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, std), len(self._shape))
    if self._dist == 'huber':
      return tfd.Independent(
          tools.UnnormalizedHuber(mean, std, 1.0), len(self._shape))
    if self._dist == 'binary':
      return tfd.Independent(tfd.Bernoulli(mean), len(self._shape))
    if self._dist == 'sigmoid':
      return tf.sigmoid(mean)
    if self._dist == 'none':
      return mean
    raise NotImplementedError(self._dist)

class ActionHead(tools.Module):

  def __init__(
      self, size, layers, units, act=tf.nn.elu, dist='trunc_normal',
      init_std=0.0, min_std=0.1, action_disc=5, temp=0.1, outscale=0):
    # assert min_std <= 2
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._action_disc = action_disc
    self._temp = temp() if callable(temp) else temp
    self._outscale = outscale

  def __call__(self, features, dtype=None):
    # print (' features ', features.shape, features.shape[:-3])
    x = tools._convert_dense(features)
    # print ("in action head ", x.shape)
    for index in range(self._layers):
      kw = {}
      if index == self._layers - 1 and self._outscale:
        kw['kernel_initializer'] = tf.keras.initializers.VarianceScaling(
            self._outscale)
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act, **kw)(x)
    if self._dist == 'tanh_normal':
      # https://www.desmos.com/calculator/rcmcf5jwe7
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      mean = tf.tanh(mean)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'tanh_normal_5':
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      mean = 5 * tf.tanh(mean / 5)
      std = tf.nn.softplus(std + 5) + 5
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
      # print ()
    elif self._dist == 'normal':
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      mean, std = tf.split(x, 2, -1)
      std = tf.nn.softplus(std + self._init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'normal_1':
      mean = self.get(f'hout', tfkl.Dense, self._size)(x)
      if dtype:
        mean = tf.cast(mean, dtype)
      dist = tfd.Normal(mean, 1)
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'trunc_normal':
      # https://www.desmos.com/calculator/mmuvuhnyxo
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      x = tf.cast(x, tf.float32)
      mean, std = tf.split(x, 2, -1)
      mean = tf.tanh(mean)
      std = 2 * tf.nn.sigmoid(std / 2) + self._min_std
      dist = tools.SafeTruncatedNormal(mean, std, -1, 1)
      dist = tools.DtypeDist(dist, dtype)
      dist = tfd.Independent(dist, 1)
    elif self._dist == 'onehot':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      x = tf.cast(x, tf.float32)
      dist = tools.OneHotDist(x, dtype=dtype)
      dist = tools.DtypeDist(dist, dtype)
    elif self._dist == 'onehot_gumble':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      if dtype:
        x = tf.cast(x, dtype)
      temp = self._temp
      dist = tools.GumbleDist(temp, x, dtype=dtype)
    else:
      raise NotImplementedError(self._dist)
    return dist

class GRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, size, norm=False, act=tf.tanh, update_bias=-1, **kwargs):
    super().__init__()
    self._size = size
    self._act = act
    self._norm = norm
    self._update_bias = update_bias
    self._layer = tfkl.Dense(3 * size, use_bias=norm is not None, **kwargs)
    if norm:
      self._norm = tfkl.LayerNormalization(dtype=tf.float32)

  @property
  def state_size(self):
    return self._size

  def call(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    parts = self._layer(tf.concat([inputs, state], -1))
    if self._norm:
      dtype = parts.dtype
      parts = tf.cast(parts, tf.float32)
      parts = self._norm(parts)
      parts = tf.cast(parts, dtype)
    reset, cand, update = tf.split(parts, 3, -1)
    reset = tf.nn.sigmoid(reset)
    cand = self._act(reset * cand)
    update = tf.nn.sigmoid(update + self._update_bias)
    output = update * cand + (1 - update) * state
    return output, [output]

class ConvGRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, spatial=6, depth=32, kernel=3):
    super().__init__()
    self._depth = depth
    self._kernel = kernel
    self._spatial = spatial
    self._reset = tfkl.Conv2D(self._depth, self._kernel, padding='same', kernel_initializer=tf.keras.initializers.Orthogonal())
    self._update = tfkl.Conv2D(self._depth, self._kernel, padding='same', kernel_initializer=tf.keras.initializers.Orthogonal())
    self._out = tfkl.Conv2D(self._depth, self._kernel, padding='same', kernel_initializer=tf.keras.initializers.Orthogonal())

  @property
  def state_size(self):
      return tf.TensorShape([self._spatial, self._spatial, self._depth])

  def __call__(self, inputs, state):
    state = state[0]  # Keras wraps the state in a list.
    stacked_inputs = tf.concat([inputs, state], -1)

    update = tf.nn.sigmoid(self._update(stacked_inputs))
    reset = tf.nn.sigmoid(self._reset(stacked_inputs))
    cand = tf.concat([inputs, state * reset], -1)
    cand = tf.tanh(self._out(cand))
    output = update * cand + (1 - update) * state

    return output, [output]

class StochasticConvGRUCell(tf.keras.layers.AbstractRNNCell):

  def __init__(self, spatial=8, depth=32, kernel=5, activation=tf.tanh, recurrent_activation=tf.nn.sigmoid, skip=True):
    super().__init__()
    self._depth = depth
    self._kernel = kernel
    self._spatial = spatial
    self._embed_dim=8
    self._reset = tfkl.Conv2D(self._depth, self._kernel, padding='same', kernel_initializer=tf.keras.initializers.Orthogonal(), use_bias=True, bias_initializer=tf.keras.initializers.Ones())
    self._update = tfkl.Conv2D(self._depth, self._kernel, padding='same', kernel_initializer=tf.keras.initializers.Orthogonal(), use_bias=True, bias_initializer=tf.keras.initializers.Ones())
    self._out = tfkl.Conv2D(self._depth, self._kernel, padding='same', kernel_initializer=tf.keras.initializers.Orthogonal(), use_bias=True, bias_initializer='zeros')
    self._update_u1 = tfkl.Dense(self._embed_dim)
    self._update_u2 = tfkl.Dense(self._depth)
    self._activation = activation
    self._recurrent_activation = recurrent_activation
    self._skip = skip

  @property
  def state_size(self):
    return tf.TensorShape([self._spatial, self._spatial, self._depth])

  def call(self, inputs, states):
    state, update_sample, update_prob, _ = states

    stacked_inputs = tf.concat([inputs, state], -1)
    update = self._recurrent_activation(self._update(stacked_inputs))
    reset = self._recurrent_activation(self._reset(stacked_inputs))
    cand = self._activation(self._out(tf.concat([inputs, state * reset], -1)))
    new_state_tilde = update * cand + (1 - update) * state

    # new_update_prob_tilde = self._recurrent_activation(self._update_prob_conv(new_state_tilde))
    # cum_update_prob = cum_update_prob_prev + tf.minimum(update_prob_prev, 1. - cum_update_prob_prev)
    # update_gate = tools.BernoulliDist(probs=cum_update_prob).sample()

    n_dims = len(new_state_tilde.shape)
    update_prob_tilde_ = self._update_u1(tf.reshape(tf.transpose(new_state_tilde, perm=list(range(n_dims-3)) + [n_dims-1, n_dims-3, n_dims-2]), list(new_state_tilde.shape[:-3]) + [self._depth, self._spatial * self._spatial]))
    update_prob_tilde  = self._recurrent_activation(self._update_u2(tf.reshape(update_prob_tilde_, list(new_state_tilde.shape[:-3]) + [self._depth * self._embed_dim])))

    # print (' stochatic, conv gtu cell:', state.shape, update_prob_tilde)
    if self._skip:
      new_update_prob = update_sample * update_prob_tilde + (1. - update_sample) * (update_prob + tf.minimum(1.0 - update_prob, update_prob_tilde))
    else:
      print ('no skip connection')
      new_update_prob = update_prob_tilde

    new_update_sample = tfd.Independent(tools.BernoulliDist(probs=new_update_prob), 1).sample()

    new_state = tools._convert_conv(new_update_sample, self._spatial) * new_state_tilde + (1.0 - tools._convert_conv(new_update_sample, self._spatial)) * state
    new_output = new_state

    return new_output, [new_state, new_update_sample, new_update_prob, update_prob_tilde]

class SelfAttnModel(tf.keras.Model):

  def __init__(self, input_dims, **kwargs):
    super(SelfAttnModel, self).__init__(**kwargs)
    self.attn = _Attention()
    self.query_conv = tf.keras.layers.Conv2D(filters=input_dims // 8, kernel_size=1)
    self.key_conv = tf.keras.layers.Conv2D(filters=input_dims // 8, kernel_size=1)
    self.value_conv = tf.keras.layers.Conv2D(filters=input_dims, kernel_size=1)

  def call(self, inputs):
    q = self.query_conv(inputs)
    k = self.key_conv(inputs)
    v = self.value_conv(inputs)
    return self.attn([q, k, v, inputs])

class _Attention(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super(_Attention, self).__init__(**kwargs)

  def build(self, input_shapes):
    self.gamma = self.add_weight(self.name + '_gamma', shape=(), initializer=tf.initializers.Zeros)

  def call(self, inputs):
    if len(inputs) != 4:
        raise Exception('an attention layer should have 4 inputs')

    query_tensor = inputs[0]
    key_tensor =  inputs[1]
    value_tensor = inputs[2]
    origin_input = inputs[3]

    input_shape = tf.shape(query_tensor)

    height_axis = 1
    width_axis = 2

    batchsize = input_shape[0]
    height = input_shape[height_axis]
    width = input_shape[width_axis]

    proj_query = tf.reshape(query_tensor, (batchsize, height*width, -1))
    proj_key = tf.transpose(tf.reshape(key_tensor, (batchsize, height*width, -1)), (0, 2, 1))
    proj_value = tf.transpose(tf.reshape(value_tensor, (batchsize, height*width, -1)), (0, 2, 1))

    energy = tf.matmul(proj_query, proj_key)
    attention = tf.nn.softmax(energy)
    out = tf.matmul(proj_value, tf.transpose(attention, (0, 2, 1)))

    out = tf.reshape(tf.transpose(out, (0, 2, 1)), (batchsize, height, width, -1))

    return tf.add(tf.multiply(out, self.gamma), origin_input)

class SlotAttention(tools.Module):
  """Slot Attention module."""

  def __init__(self, spatial_size, #mlp_hidden_size,
               epsilon=1e-8):
    """Builds the Slot Attention module.
    Args:
      num_iterations: Number of iterations.
      num_slots: Number of slots.
      slot_size: Dimensionality of slot feature vectors.
      mlp_hidden_size: Hidden layer size of MLP.
      epsilon: Offset for attention coefficients before normalization.
    """
    # super().__init__()
    # self.num_slots = num_slots
    self.slot_spatial_size = spatial_size
    self.slot_size = spatial_size * spatial_size
    # self.mlp_hidden_size = mlp_hidden_size
    self.epsilon = epsilon

    self.norm_inputs = tfkl.LayerNormalization(dtype=tf.float32) #layers.LayerNormalization()
    # self.norm_slots = layers.LayerNormalization()
    # self.norm_mlp = layers.LayerNormalization()

    # Linear maps for the attention module.
    self.project_q = tfkl.Dense(self.slot_size, use_bias=False, kernel_initializer=tf.keras.initializers.Identity()) #layers.Dense(self.slot_size, use_bias=False, name="q")
    self.project_k = tfkl.Dense(self.slot_size, use_bias=False, kernel_initializer=tf.keras.initializers.Identity()) #layers.Dense(self.slot_size, use_bias=False, name="k")
    self.project_v = tfkl.Dense(self.slot_size, use_bias=False, kernel_initializer=tf.keras.initializers.Identity()) #layers.Dense(self.slot_size, use_bias=False, name="v")

    # Slot update functions.
    # self.mlp = tf.keras.Sequential([
    #     layers.Dense(self.mlp_hidden_size, activation="relu"),
    #     layers.Dense(self.slot_size)
    # ], name="mlp")

  def __call__(self, inputs):
    # return inputs
    # `inputs` has shape [batch_size, num_inputs, inputs_size].
    input_shape = inputs.shape
    n_dims = len(input_shape)
    # print (input_shape, tf.transpose(inputs, perm=list(range(n_dims-3)) + [n_dims-1, n_dims-3, n_dims-3]).shape)
    inputs = tf.transpose(inputs, perm=list(range(len(inputs.shape[:-3]))) + [n_dims-1, n_dims-3, n_dims-2])
    inputs = tf.reshape(inputs, list(inputs.shape[:-2]) + [self.slot_spatial_size * self.slot_spatial_size])

    inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
    k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
    v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

    # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
    # slots = self.slots_mu + tf.exp(self.slots_log_sigma) * tf.random.normal(
    #     [tf.shape(inputs)[0], self.num_slots, self.slot_size])

    # Multiple rounds of attention.
    # for _ in range(self.num_iterations):
      # slots_prev = slots
      # slots = self.norm_slots(slots)

    # Attention.
    q = self.project_q(inputs)  # Shape: [batch_size, num_slots, slot_size].
    q *= self.slot_size ** -0.5  # Normalization.
    attn_logits = tf.keras.backend.batch_dot(k, q, axes=-1)
    attn = tf.nn.softmax(attn_logits, axis=-1)
    # `attn` has shape: [batch_size, num_inputs, num_slots].

    # Weigted mean.
    attn += self.epsilon
    attn /= tf.reduce_sum(attn, axis=-2, keepdims=True)
    updates = tf.keras.backend.batch_dot(attn, v, axes=-2)
      # `updates` has shape: [batch_size, num_slots, slot_size].

      # # Slot update.
      # slots, _ = self.gru(updates, [slots_prev])
      # slots += self.mlp(self.norm_mlp(slots))
    updates = tf.reshape(updates, list(updates.shape[:-1]) + [self.slot_spatial_size, self.slot_spatial_size])
    updates = tf.transpose(updates, list(range(n_dims-3)) + [n_dims-2, n_dims-1, n_dims-3])
    return updates
