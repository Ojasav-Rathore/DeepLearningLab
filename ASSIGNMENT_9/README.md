# Generative Adversarial Networks (GANs): Model Variants and Experimental Analysis

**Weights & Biases Dashboard:** [https://wandb.ai/ojasavrathore_25afi13-delhi-technological-university/exp9-gans?nw=nwuserojasavrathore_25afi13]  
**Hugging Face Model Repository:** [https://huggingface.co/ojasav-rathore/experiment-9-models]  

## Project Context
This repository implements and evaluates Generative Adversarial Networks (GANs) for image generation. Mastering these foundational generative models and their underlying training dynamics is a critical stepping stone for advanced research in AI security, assessing adversarial attack vulnerabilities, and developing robust deepfake detection architectures. 

## Objective
The primary goal is to implement a GAN and analyze its performance under varying configurations. The project evaluates the effect of model architectures (Vanilla GAN vs. DCGAN), the impact of different loss functions, and the influence of optimizers on training stability and convergence.

## Dataset
* **Source:** Fashion-MNIST
* **Format:** $28\times28$ grayscale images of clothing items 
* **Preprocessing:** Images are normalized to the range [-1, 1].

## Architectures Implemented
1.  **Vanilla GAN:** Utilizes fully connected dense layers for both the generator and the discriminator.
2. **DCGAN (Deep Convolutional GAN):** Employs transposed convolutions, Batch Normalization, and ReLU activations in the generator network. The discriminator uses standard convolutional layers paired with LeakyReLU activations.

## Experimental Results

The models were trained using alternating adversarial training. Configurations tested included Binary Cross-Entropy (BCE), Least Squares GAN (LSGAN), and Wasserstein Loss (WGAN) , paired with SGD, RMSprop, and Adam optimizers.

| Configuration | Generator Loss | Discriminator Loss |
| :--- | :--- | :--- |
| vanilla_wgan_rmsprop | -0.9214 | -0.4561 |
| dcgan_wgan_rmsprop | -0.0057 | -0.0146 |
| vanilla_wgan_adam | -0.0007 | -0.1695 |
| dcgan_wgan_adam | 0.0032 | -0.0054 |
| dcgan_wgan_sgd | 0.2844 | -0.6060 |
| vanilla_lsgan_adam | 0.5509 | 0.1731 |
| vanilla_wgan_sgd | 0.5887 | -1.9224 |
| vanilla_lsgan_rmsprop| 0.9315 | 0.0839 |
| vanilla_lsgan_sgd | 0.9658 | 0.0202 |
| dcgan_lsgan_sgd | 1.0036 | 0.0029 |
| dcgan_lsgan_rmsprop | 1.0248 | 0.0149 |
| dcgan_lsgan_adam | 1.0390 | 0.0166 |
| vanilla_bce_adam | 1.3835 | 0.5314 |
| dcgan_bce_adam | 1.3870 | 0.4230 |
| dcgan_bce_rmsprop | 1.5786 | 0.3214 |
| vanilla_bce_rmsprop | 2.9361 | 0.2276 |
| vanilla_bce_sgd | 3.0613 | 0.1533 |
| dcgan_bce_sgd | 9.5567 | 0.0001 |

## Analysis and Discussion

### GAN vs DCGAN
* **Convolution vs. Dense:** DCGAN vastly improves visual quality because convolutional layers maintain and leverage spatial relationships in the images, unlike Vanilla GANs which flatten the spatial dimensions.
* **Role of Batch Normalization:** Incorporating batch normalization in DCGAN stabilizes the learning process by ensuring gradients are kept within a reasonable scale, preventing premature convergence.

### Loss Function Comparison
* **Stability of BCE:** BCE is highly susceptible to vanishing gradients. This is empirically proven in our `dcgan_bce_sgd` configuration, where the discriminator reached near-perfection (loss: 0.0001) causing the generator's loss to explode (9.5567) and stop learning.
* **Advantages of Wasserstein Loss:** WGAN loss correlates directly with image quality and provides reliable gradients even when the discriminator is well-trained. Our WGAN configurations maintained tightly balanced, near-zero losses, exhibiting the highest stability.

### Optimizer Comparison
* **Adam vs. SGD:** Adam is generally preferred in adversarial settings because its adaptive learning rates and momentum help navigate the highly non-convex loss landscapes of GANs.
* **Convergence Stability:** The results show that SGD often fails to maintain the delicate balance required between the generator and discriminator, leading to failure states (like in BCE), whereas Adam and RMSprop drive far more stable convergence.

### Training Challenges
Throughout the training cycles, typical generative challenges were observed:
* **Vanishing Gradients:** Seen primarily when the discriminator overpowers the generator early in training.
* **Mode Collapse:** The generator producing only a small subset of the Fashion-MNIST classes, which was effectively mitigated by switching from BCE to WGAN.
* **Oscillatory Behavior:** Tracking the loss metrics over epochs revealed continuous fluctuations, highlighting the inherent instability of the adversarial min-max game.

## Observations
* **Improvement Over Epochs:** Generator outputs typically start as pure noise and iteratively sharpen as the adversarial feedback loop forces better representations.
* **Quality vs. Diversity Trade-off:** Certain configurations pushed for sharper images but lost class representation (diversity), requiring careful tuning.
* **Hyperparameter Sensitivity:** The stark variance in the results table proves that GAN success is acutely sensitive to minor changes in learning rate, optimizer choice, and loss function formulation.
