import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

import networks
import tools


class WorldModel(tools.Module):

  def __init__(self, step, config):
    self._step = step
    self._config = config
    self.encoder = networks.ConvEncoder(
        config.cnn_depth, config.act, config.encoder_kernels)
    self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell)
    self.heads = {}
    channels = (1 if config.grayscale else 3)
    shape = config.size + (channels,)
    self.heads['image'] = networks.ConvDecoder(
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
          [], config.discount_layers, config.units, config.act, dist='binary')
    for name in config.grad_heads:
      assert name in self.heads, name
    self._model_opt = tools.Optimizer(
        'model', config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt)
    self._scales = dict(
        reward=config.reward_scale, discount=config.discount_scale)

  def train(self, data):
    data = self.preprocess(data)
    with tf.GradientTape() as model_tape:
      embed = self.encoder(data)
      post, prior = self.dynamics.observe(embed)
      kl_balance = tools.schedule(self._config.kl_balance, self._step)
      kl_free = tools.schedule(self._config.kl_free, self._step)
      kl_scale = tools.schedule(self._config.kl_scale, self._step)

      kl_loss, kl_value = self.dynamics.kl_loss(
          post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
      losses = {}
      likes = {}
      for name, head in self.heads.items():
        grad_head = (name in self._config.grad_heads)
        feat = self.dynamics.get_feat(post)
        feat = feat if grad_head else tf.stop_gradient(feat)
        pred = head(feat, tf.float32)
        like = pred.log_prob(tf.cast(data[name], tf.float32))
        likes[name] = like
        losses[name] = -tf.reduce_mean(like) * self._scales.get(name, 1.0)
      model_loss = sum(losses.values()) + kl_loss
    model_parts = [self.encoder, self.dynamics] + list(self.heads.values())
    metrics = self._model_opt(model_tape, model_loss, model_parts)
    metrics.update({f'{name}_loss': loss for name, loss in losses.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    metrics['kl'] = tf.reduce_mean(kl_value)
    metrics['prior_ent'] = self.dynamics.get_dist(prior).entropy()
    metrics['post_ent'] = self.dynamics.get_dist(post).entropy()
    context = dict(
        embed=embed, feat=self.dynamics.get_feat(post),
        kl=kl_value, postent=self.dynamics.get_dist(post).entropy())
    return post, context, metrics

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    obs['image'] = tf.cast(obs['obs'], dtype) / 255.0 - 0.5
    if 'discount' in obs:
      obs['discount'] *= self._config.discount
    for key, value in obs.items():
      if tf.dtypes.as_dtype(value.dtype) in (
          tf.float16, tf.float32, tf.float64):
        obs[key] = tf.cast(value, dtype)
    return obs

  @tf.function
  def video_pred(self, data, initial_frames=10, nenvs=6):
    data = self.preprocess(data)
    total_frames = len(data['image'][0])
    truth = data['image'][:nenvs] + 0.5
    embed = self.encoder(data)
    states, _ = self.dynamics.observe(embed[:nenvs, :])
    recon = self.heads['image'](self.dynamics.get_feat(states)).mode()[:nenvs]
    init = {k: v[:, initial_frames-1] for k, v in states.items()}
    prior = self.dynamics.imagine((nenvs, total_frames-initial_frames), init)
    openl = self.heads['image'](self.dynamics.get_feat(prior)).mode()
    model_recon = recon[:, :] + 0.5
    error_recon = (model_recon - truth + 1) / 2
    model_imagine = tf.concat([recon[:, :initial_frames] + 0.5, openl + 0.5], 1)
    error_imagine = (model_imagine - truth + 1) / 2
    return tf.concat([truth, model_imagine, error_imagine], 2), tf.concat([truth, model_recon, error_recon], 2), (truth, model_imagine), (truth, model_recon)


class ConvGRUClassifier(tools.Module):
  def __init__(self, config, world_model, task, n_classes):
    self._config = config
    self._world_model = world_model
    self._task = task
    self._cell_type = config.dyn_classifer_cell
    if config.dyn_classifer_cell == 'gru':
      print("Using GRUCell in the Classifier")
      self._cell = networks.GRUCell(config.dyn_classifer_cell_units, True)
    else:
      print("Using No Cell in the Classifier")
      self._cell = None
    self._classifier = networks.DenseHead(n_classes, config.classifer_layers, config.units, config.act, 'none')
    kw = dict(wd=config.weight_decay, opt=config.opt)
    self._classifer_opt = tools.Optimizer('classifier', config.classifier_lr, config.opt_eps, config.classifier_grad_clip, **kw)
    if self._task == 1 or self._task == 2:
      self._metric = tools.mAP(n_classes, True)
    else:
      self._metric = tf.keras.metrics.TopKCategoricalAccuracy(k=5)

  def train(self, data, labels):
    '''
    Take in data, tf.shape(n * B, H, W, C), 'labels': tf.shape(B, n_classes))
    B: Batch size, i.e (No. of videos)
    n: Chunks
    T: Frames in each chunk
    H, W, C: Height, width, channels
    '''
    with tf.GradientTape() as classifier_tape:
      if self._task == 1 or self._task == 2:
        objective = tf.nn.sigmoid_cross_entropy_with_logits
      else:
        objective = tf.nn.softmax_cross_entropy_with_logits
      data = self.reshape_data(data)
      data = self.agg_feat(data)
      preds = self._classifier(data)
      preds = tf.cast(preds, tf.float32)
      loss = objective(labels, preds)
      loss = tf.math.reduce_mean(loss)
      self._metric.update_state(labels, preds)
    if self._cell is None:
      classifier_parts = [self._classifier]
    else:
      classifier_parts = [self._cell, self._classifier]
    metrics = self._classifer_opt(classifier_tape, loss, classifier_parts)
    classifier_acc = self._metric.result()
    metrics.update({'classifier_loss': loss, 'classifier_metric': classifier_acc})
    self._metric.reset_states()
    return metrics

  def reshape_data(self, data):
    '''
    Args:
      data -> tf.shape(n*B, H, W, C)
    Returns:
      data -> tf.shape(n, B, H, W, C)
    '''
    nB = data.shape[0]
    B = self._config.batch_size
    data = tf.reshape(data, [B, nB // B] + list(data.shape[-1:]))
    data = tf.transpose(data, [1, 0] + list(range(2, len(data.shape))))
    return data

  def agg_feat(self, post, state=None):
    '''
    Args:
      post -> tf.shape(n, B, H, W, C)
    Returns:
      x -> tf.shape(B, H, W, C)
    '''
    if self._cell is not None:
      if state is None:
        state = self._cell.get_initial_state(None, post.shape[1], post.dtype)
      x = tf.zeros_like(state)
      for post_state in post:
        x, state = self._cell(post_state, [state])
        state = state[0]
    else:
      x = tf.math.reduce_mean(post, 0)
    return x

  @tf.function
  def action_pred(self, data):
    data = self._world_model.preprocess(data)
    embed = self._world_model.encoder(data)
    states, _ = self._world_model.dynamics.observe(embed)
    chunk_posteriors = self._world_model.dynamics.get_feat(states)[:, -1]
    chunk_posteriors = self.reshape_data(chunk_posteriors)
    chunk_posteriors = self.agg_feat(chunk_posteriors)
    preds = self._classifier(chunk_posteriors)
    return preds


# class ImagBehavior(tools.Module):

#   def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
#     self._config = config
#     self._world_model = world_model
#     self._stop_grad_actor = stop_grad_actor
#     self._reward = reward
#     self.actor = networks.ActionHead(
#         config.num_actions, config.actor_layers, config.units, config.act,
#         config.actor_dist, config.actor_init_std, config.actor_min_std,
#         config.actor_dist, config.actor_temp, config.actor_outscale)
#     self.value = networks.DenseHead(
#         [], config.value_layers, config.units, config.act,
#         config.value_head)
#     if config.slow_value_target or config.slow_actor_target:
#       self._slow_value = networks.DenseHead(
#           [], config.value_layers, config.units, config.act)
#       self._updates = tf.Variable(0, tf.int64)
#     kw = dict(wd=config.weight_decay, opt=config.opt)
#     self._actor_opt = tools.Optimizer(
#         'actor', config.actor_lr, config.opt_eps, config.actor_grad_clip, **kw)
#     self._value_opt = tools.Optimizer(
#         'value', config.value_lr, config.opt_eps, config.value_grad_clip, **kw)

#   def train(
#       self, start, objective=None, imagine=None, tape=None, repeats=None):
#     objective = objective or self._reward
#     self._update_slow_target()
#     metrics = {}
#     with (tape or tf.GradientTape()) as actor_tape:
#       assert bool(objective) != bool(imagine)
#       if objective:
#         imag_feat, imag_state, imag_action = self._imagine(
#             start, self.actor, self._config.imag_horizon, repeats)
#         reward = objective(imag_feat, imag_state, imag_action)
#       else:
#         imag_feat, imag_state, imag_action, reward = imagine(start)
#       actor_ent = self.actor(imag_feat, tf.float32).entropy()
#       state_ent = self._world_model.dynamics.get_dist(
#           imag_state, tf.float32).entropy()
#       target, weights = self._compute_target(
#           imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
#           self._config.slow_actor_target)
#       actor_loss, mets = self._compute_actor_loss(
#           imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
#           weights)
#       metrics.update(mets)
#     if self._config.slow_value_target != self._config.slow_actor_target:
#       target, weights = self._compute_target(
#           imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
#           self._config.slow_value_target)
#     value_input = imag_feat
#     with tf.GradientTape() as value_tape:
#       value = self.value(value_input, tf.float32)[:-1]
#       value_loss = -value.log_prob(tf.stop_gradient(target))
#       if self._config.value_decay:
#         value_loss += self._config.value_decay * value.mode()
#       value_loss = tf.reduce_mean(weights[:-1] * value_loss)
#     metrics['reward_mean'] = tf.reduce_mean(reward)
#     metrics['reward_std'] = tf.math.reduce_std(reward)
#     metrics['actor_ent'] = tf.reduce_mean(actor_ent)
#     metrics.update(self._actor_opt(actor_tape, actor_loss, [self.actor]))
#     metrics.update(self._value_opt(value_tape, value_loss, [self.value]))
#     return imag_feat, imag_state, imag_action, weights, metrics

#   def _imagine(self, start, policy, horizon, repeats=None):
#     dynamics = self._world_model.dynamics
#     if repeats:
#       start = {k: tf.repeat(v, repeats, axis=1) for k, v in start.items()}
#     flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
#     start = {k: flatten(v) for k, v in start.items()}
#     def step(prev, _):
#       state, _, _ = prev
#       feat = dynamics.get_feat(state)
#       inp = tf.stop_gradient(feat) if self._stop_grad_actor else feat
#       action = policy(inp).sample()
#       succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
#       return succ, feat, action
#     feat = 0 * dynamics.get_feat(start)
#     action = policy(feat).mode()
#     succ, feats, actions = tools.static_scan(
#         step, tf.range(horizon), (start, feat, action))
#     states = {k: tf.concat([
#         start[k][None], v[:-1]], 0) for k, v in succ.items()}
#     if repeats:
#       def unfold(tensor):
#         s = tensor.shape
#         return tf.reshape(tensor, [s[0], s[1] // repeats, repeats] + s[2:])
#       states, feats, actions = tf.nest.map_structure(
#           unfold, (states, feats, actions))
#     return feats, states, actions

#   def _compute_target(
#       self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
#       slow):
#     reward = tf.cast(reward, tf.float32)
#     if 'discount' in self._world_model.heads:
#       inp = self._world_model.dynamics.get_feat(imag_state)
#       discount = self._world_model.heads['discount'](inp, tf.float32).mean()
#     else:
#       discount = self._config.discount * tf.ones_like(reward)
#     if self._config.future_entropy and tf.greater(
#         self._config.actor_entropy(), 0):
#       reward += self._config.actor_entropy() * actor_ent
#     if self._config.future_entropy and tf.greater(
#         self._config.actor_state_entropy(), 0):
#       reward += self._config.actor_state_entropy() * state_ent
#     if slow:
#       value = self._slow_value(imag_feat, tf.float32).mode()
#     else:
#       value = self.value(imag_feat, tf.float32).mode()
#     target = tools.lambda_return(
#         reward[:-1], value[:-1], discount[:-1],
#         bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
#     weights = tf.stop_gradient(tf.math.cumprod(tf.concat(
#         [tf.ones_like(discount[:1]), discount[:-1]], 0), 0))
#     return target, weights

#   def _compute_actor_loss(
#       self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
#       weights):
#     metrics = {}
#     inp = tf.stop_gradient(imag_feat) if self._stop_grad_actor else imag_feat
#     policy = self.actor(inp, tf.float32)
#     actor_ent = policy.entropy()
#     if self._config.imag_gradient == 'dynamics':
#       actor_target = target
#     elif self._config.imag_gradient == 'reinforce':
#       imag_action = tf.cast(imag_action, tf.float32)
#       actor_target = policy.log_prob(imag_action)[:-1] * tf.stop_gradient(
#           target - self.value(imag_feat[:-1], tf.float32).mode())
#     elif self._config.imag_gradient == 'both':
#       imag_action = tf.cast(imag_action, tf.float32)
#       actor_target = policy.log_prob(imag_action)[:-1] * tf.stop_gradient(
#           target - self.value(imag_feat[:-1], tf.float32).mode())
#       mix = self._config.imag_gradient_mix()
#       actor_target = mix * target + (1 - mix) * actor_target
#       metrics['imag_gradient_mix'] = mix
#     else:
#       raise NotImplementedError(self._config.imag_gradient)
#     if not self._config.future_entropy and tf.greater(
#         self._config.actor_entropy(), 0):
#       actor_target += self._config.actor_entropy() * actor_ent[:-1]
#     if not self._config.future_entropy and tf.greater(
#         self._config.actor_state_entropy(), 0):
#       actor_target += self._config.actor_state_entropy() * state_ent[:-1]
#     actor_loss = -tf.reduce_mean(weights[:-1] * actor_target)
#     return actor_loss, metrics

#   def _update_slow_target(self):
#     if self._config.slow_value_target or self._config.slow_actor_target:
#       if self._updates % self._config.slow_target_update == 0:
#         mix = self._config.slow_target_fraction
#         for s, d in zip(self.value.variables, self._slow_value.variables):
#           d.assign(mix * s + (1 - mix) * d)
#       self._updates.assign_add(1)
