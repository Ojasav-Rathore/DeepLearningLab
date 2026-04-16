
# Experiment 8: Autoencoders and Variational Autoencoders (VAE) with Latent Space Analysis

##  Project Overview
This repository contains the PyTorch implementation of Autoencoders (AE) and Variational Autoencoders (VAE) evaluated on the Fashion-MNIST dataset. The objective of this experiment is to learn latent representations, compare deterministic versus probabilistic generative models, and analyze their behavior under various hyperparameters (latent dimensions, loss functions, and optimizers).

##  Live Tracking & Models
* **Weights & Biases Dashboard:** [View Full W&B Report](https://wandb.ai/ojasavrathore_25afi13-delhi-technological-university/experiment-8-ae-vae)
* **Hugging Face Repository:** [Trained Models & Outputs](https://huggingface.co/ojasav-rathore/experiment-8-models)

##  Dataset & Preprocessing
* **Dataset:** Fashion-MNIST (Grayscale images of clothing items).
* **Split:** 80% Training | 10% Validation | 10% Testing.
* **Preprocessing:** Images flattened to 1D arrays (784 pixels) and normalized to a `[0, 1]` scale.

---

##  Results & Hyperparameter Analysis

An extensive grid search was conducted across 48 configurations running for 5 epochs each. The configurations tested combinations of:
* **Model Type:** Autoencoder (AE) vs. Variational Autoencoder (VAE)
* **Latent Dimensions:** 2, 8, 16, 32
* **Loss Functions:** Binary Cross-Entropy (BCE) vs. Mean Squared Error (MSE)
* **Optimizers:** Adam, RMSprop, SGD

### 1. Autoencoder vs. VAE (Deterministic vs. Probabilistic)
The logs demonstrate a clear trade-off between exact reconstruction and generative capability:
* **Reconstruction Dominance:** The Standard Autoencoder consistently achieved lower reconstruction loss compared to the VAE under identical configurations. For example, `AE_dim32_MSE_Adam` achieved a validation loss of **8.01**, while `VAE_dim32_MSE_Adam` plateaued at **26.38**. 
* **The VAE Trade-off:** The higher loss in VAEs is expected. The VAE optimizes not just for reconstruction, but also incorporates a **KL Divergence** penalty to force the latent space to approximate a standard normal distribution. This regularization prevents overfitting and ensures a smooth, continuous latent space required for generating new, meaningful data interpolations.

### 2. Effect of Latent Space Dimensionality
Increasing the latent dimension size directly improved model performance by reducing the bottleneck constraint, allowing more information to flow to the decoder.
* **Dim 2:** Severely restricted capacity. (`AE_dim2_MSE_Adam` Val Loss: 23.44)
* **Dim 8:** Significant improvement. (`AE_dim8_MSE_Adam` Val Loss: 11.97)
* **Dim 16:** Diminishing returns begin. (`AE_dim16_MSE_Adam` Val Loss: 9.60)
* **Dim 32:** Best reconstruction quality. (`AE_dim32_MSE_Adam` Val Loss: 8.01)

While dimension 32 yields the sharpest image reconstruction, dimension 2 allows for direct 2D coordinate visualization of the latent space clusters.

### 3. Optimizer Comparison
Adam drastically outperformed the other optimizers in both convergence speed and final loss values across all models and configurations.
* **Adam:** Fastest convergence, finding stable minima within the first 2-3 epochs.
* **RMSprop:** Performed similarly to Adam but consistently trailed slightly behind in final validation loss (e.g., `AE_dim32_MSE_RMSprop` Val Loss: 9.32).
* **SGD:** Struggled severely. Without momentum tuning, standard SGD failed to navigate the complex loss landscapes efficiently, resulting in massive final losses (e.g., `AE_dim32_MSE_SGD` Val Loss: 71.37). 

### 4. Loss Function Behavior (BCE vs. MSE)
*Note: BCE and MSE losses exist on different mathematical scales, so their absolute values cannot be directly compared (BCE is heavily penalized by probabilities, MSE by squared distances).*
* **MSE:** Optimized for overall pixel-level distance, generally leading to slightly blurrier reconstructions but stable numerical training.
* **BCE:** Because inputs were normalized to `[0,1]`, BCE effectively treats pixel intensities as probabilities. This generally results in sharper contrasts for edge details in the Fashion-MNIST dataset, though it is more susceptible to gradient instability if predictions hit exactly 0 or 1.

---

##  Key Observations & Conclusion

1.  **Reconstruction vs. Generation Trade-off:** Autoencoders are excellent for strict data compression and denoising, but their latent spaces are disjointed. VAEs sacrifice some reconstruction sharpness to create a well-regulated, probabilistic latent space capable of smooth interpolations (e.g., morphing a sneaker into a boot).
2.  **Capacity Bottlenecks:** A latent dimension of 2 forces too much data loss for complex items like clothing. A dimension of 16-32 provides a much better balance of compression and fidelity.
3.  **Optimization:** Adaptive learning rate algorithms (Adam, RMSprop) are strictly necessary for quickly training these generative models. Standard SGD is too slow and gets stuck in local minima.

---
