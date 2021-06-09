# ODE-RL

Using <a href="https://arxiv.org/abs/1806.07366">Neural Ordinary Differential Equations</a> to model continuous time dynamics for sample-efficient Reinforcement Learning

## Dataset generation

### Moving MNIST

- Download the MNIST file <b>train-images-idx3-ubyte.gz</b> from <a href="http://yann.lecun.com/exdb/mnist/">here</a> and move it to where you store your datasets (eg. ~/scratch/datasets/MovingMNIST/train-images-idx3-ubyte.gz).

## Models

1. ConvGRU   
   - `ConvEncode` input images
   - `ConvGRUCell` uses encoded images and previous hidden state to obtain future hidden states 
   - `ConvDecode` from latent space to pixel space.  
   - Moving MNIST training and testing:
   ```
     python main.py --config defaults train_mmnist_cgru 
     python main.py --config defaults test_mmnist_cgru
   ```

2. ODEConv 
    - `ConvEncode` input images 
    - `ODEConvGRUCell` uses encoded inputs, input timesteps, and `ode_encoder_func` to find z0 in latent space
    - Use `diffeq_solver` to solve the Initial Value Problem: Given z<sub>0</sub> and t<sub>i</sub>....t<sub>i+n</sub>, use Neural ODE Decoder (`ode_decoder_func`) to predict z<sub>i</sub>...z<sub>i+n</sub> in latent space 
    - `ConvDecode` z<sub>i</sub>...z<sub>i+n</sub> to pixel space
    - Moving MNIST training:
    ```
     python main.py --config defaults train_mmnist_odecgru 
     <test_command_and_config_yet_to_be_added>
    ```




