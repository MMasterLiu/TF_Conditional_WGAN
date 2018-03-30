# TF_Conditional_WGAN
simple implementation of Conditional WGAN-GP for tensorflow.
* Wasserstein GAN:https://arxiv.org/abs/1701.07875
* WGAN-GP:https://arxiv.org/abs/1704.00028
* Conditional GAN:https://arxiv.org/abs/1411.1784

conditioning is done by just concatenating one-hot labels to inputs of generator and discriminator (critic).

just run main.py.
```bash
python3 main.py
```

see main.py for the hyperparameters.

I run this codes on python3.6.1 with tensorflow1.4.1
