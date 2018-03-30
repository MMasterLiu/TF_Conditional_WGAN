# TF_Conditional_WGAN
simple implementation of conditional wgan for tensorflow.

conditioning is done by just concatenating one-hot labels to inputs of generator and discriminator (critic).

just run main.py.
```bash
python3 main.py
```

see main.py for the hyperparameters.

I run this codes on python3.6.1 with tensorflow1.4.1
