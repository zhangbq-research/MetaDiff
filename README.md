# MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning
This repository contains the code for the paper:
<br>
[**c**](https://arxiv.org/pdf/2307.16424.pdf)
<br>
Baoquan Zhang, Chuyao Luo, Demin Yu, Huiwei Lin, Xutao Li, Yunming Ye, Bowen Zhang
<br>
AAAI 2024

### Abstract

Equipping a deep model the abaility of few-shot learning, i.e., learning quickly from only few examples, is a core challenge for artificial intelligence. Gradient-based meta-learning approaches effectively address the challenge by learning how to learn novel tasks. Its key idea is learning a deep model in a bi-level optimization manner, where the outer-loop process learns a shared gradient descent algorithm (i.e., its hyperparameters), while the inner-loop process leverage it to optimize a task-specific model by using only few labeled data. Although these existing methods have shown superior performance, the outer-loop process requires calculating second-order derivatives along the inner optimization path, which imposes considerable memory burdens and the risk of vanishing gradients. Drawing inspiration from recent progress of diffusion models, we find that the inner-loop gradient descent process can be actually viewed as a reverse process (i.e., denoising) of diffusion where the target of denoising is model weights but the origin data. Based on this fact, in this paper, we propose to model the gradient descent optimizer as a diffusion model and then present a novel task-conditional diffusion-based meta-learning, called MetaDiff, that effectively models the optimization process of model weights from Gaussion noises to target weights in a denoising manner. Thanks to the training efficiency of diffusion models, our MetaDiff do not need to differentiate through the inner-loop path such that the memory burdens and the risk of vanishing gradients can be effectvely alleviated. Experiment results show that our MetaDiff outperforms the state-of-the-art gradient-based meta-learning family in few-shot learning tasks.

### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{zhang2022metadiff,
	author    = {Zhang, Baoquan and Luo, Chuyao and Yu, Demin and Lin, Huiwei and Li, Xutao and Ye, Yunming and Zhang, Bowen},
	title     = {MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning},
	booktitle = {AAAI},
	year      = {2024},
}
```
