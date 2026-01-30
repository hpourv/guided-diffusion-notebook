# guided-diffusion

This is the codebase for [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233).

This repository is based on [openai/improved-diffusion](https://github.com/openai/improved-diffusion), with modifications for classifier conditioning and architecture improvements.

# Sampling from pre-trained diffusion model
This repository provides a ready-to-run Google Colab notebook for sampling from the guided-diffusion pre-trained diffusion model. You can sample from diffusion models trained on datasets such as FFHQ, CelebA-HQ, AFHQ, and CUB, without manually downloading the checkpoints.

# Image Restoration with pre-trained diffusion model
In recent years, numerous methods have been introduced for unsupervised image restoration using diffusion models. In this part, I aim to implement these methods based on the descriptions provided in the original papers, rather than relying on their official GitHub implementations.
