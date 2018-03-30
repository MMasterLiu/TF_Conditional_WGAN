# TF_Conditional_WGAN
simple implementation of Conditional WGAN-GP for tensorflow.
* DCGAN:https://arxiv.org/abs/1511.06434
* Wasserstein GAN:https://arxiv.org/abs/1701.07875
* WGAN-GP:https://arxiv.org/abs/1704.00028
* Conditional GAN:https://arxiv.org/abs/1411.1784

# Model
* conditioning is done by just concatenating one-hot labels to inputs of generator and discriminator (critic).
* architectures of generator and discriminator are almost same as DCGAN.

# How to run
just run main.py.
```bash
python3 main.py
```
I run this codes on python3.6.1 with tensorflow1.4.1

# Output
here are samples at iter=7000 and iter=0to1500.

![iter7000](https://raw.githubusercontent.com/yufuinn/TF_Conditional_WGAN/master/sample00007000.png "iter7000.png")
![traning_process](https://raw.githubusercontent.com/yufuinn/TF_Conditional_WGAN/master/sample0to1500.gif "sample0to1500.gif")

