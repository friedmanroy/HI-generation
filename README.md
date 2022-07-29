# Conditional Glow for HI Map Generation
This repository contains the implementation of a conditional [Glow](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf) (cGlow) used in order to model neutral hydrogen (HI) maps. While primarily focused on HI maps, the code in this repository can be used for any task where a cGlow model, conditioned on few parameters, is needed.

## Making Glow Conditional
The standard [Glow](https://d4mucfpksywv.cloudfront.net/research-covers/glow/paper/glow.pdf) architecture can be summarized by the following schematic:
![Glow architecture](assets/glow_architecture.png)
Where, for all $i$, the prior of the latent variables is defined as $y_i\sim\mathcal{N}\left(0,I\right)$.

Glow is mainly constructed by 3 types of layers:
* **Actnorm**: a channel-wise affine transformation
* **Invertible convolutions**: the core of Glow is the addition of invertible convolutions with 1x1 spatial resolution kernels. These layers mix the information in the channel dimension, while also allowing some scaling
* **Affine coupling**: the affine coupling layer is the main workhorse of the model. In Glow, the affine coupling is a neural network that uses half of the channels in order to define an affine transformation over the remaining channels

As defined, the 3 layers are invertible, which allows for training of the model using standard MLE through the use of the change of variable identity. Let $p_y(y)=\mathcal{N}\left(0, I\right)$ and $x=f^{-1}\_\theta(y)$ where the function $f\_\theta(\cdot)$ is invertible. Then, given a dataset of $x$s, the MLE is:
$$p\_x(x)=p\_y\left(f\_\theta(x)\right)\cdot |\text{det}\frac{\partial f\_\theta(x)}{\partial x}|\longrightarrow \hat{\theta}=\arg \min\_\theta \\{-\log p\_x(x)\\}$$
