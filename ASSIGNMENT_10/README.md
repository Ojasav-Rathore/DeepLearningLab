
# Experiment 10: Image Classification using Vision Transformers (ViT) with Augmentation and Model Comparison

## Objective
The goal of this assignment is to implement an image classification model using the Vision Transformer (ViT) architecture and compare its performance with a Convolutional Neural Network (ResNet-18). 

Unlike CNNs, which rely on convolution operations, Vision Transformers process images as sequences of patches and use self-attention mechanisms to learn global relationships.

## Dataset Details
* **Dataset:** CIFAR-10
* **Split:** 80% Training, 10% Validation, 10% Testing
* **Preprocessing:** Normalization and Resizing
* **Augmentations Executed:** Horizontal Flip, Vertical Flip

## Links & Repositories
* **Weights & Biases (W&B) Dashboard:** `[https://wandb.ai/ojasavrathore_25afi13-delhi-technological-university/exp10-vit-resnet?nw=nwuserojasavrathore_25afi13]`
* **Hugging Face Model Repository:** `[https://huggingface.co/ojasav-rathore/exp10-vit-resnet]`

---

## Results

Below is the evaluation of the models based on test accuracy and test loss across various training configurations.

| Configuration | Test Accuracy | Test Loss |
| :--- | :--- | :--- |
| **resnet18_ce_adam_no_aug** | **0.8019** | **0.6109** |
| resnet18_label_smooth_adam_no_aug | 0.7763 | 1.0255 |
| resnet18_focal_adam_no_aug | 0.7577 | 0.4074 |
| resnet18_label_smooth_rmsprop_no_aug | 0.7266 | 1.1354 |
| resnet18_ce_rmsprop_no_aug | 0.7125 | 0.8487 |
| resnet18_label_smooth_adam_aug | 0.7029 | 1.1755 |
| resnet18_label_smooth_sgd_no_aug | 0.7007 | 1.2281 |
| resnet18_focal_adam_aug | 0.6967 | 0.4841 |
| resnet18_ce_sgd_no_aug | 0.6956 | 0.8779 |
| resnet18_focal_sgd_no_aug | 0.6723 | 0.5946 |
| resnet18_ce_sgd_aug | 0.6707 | 0.9309 |
| resnet18_ce_adam_aug | 0.6678 | 0.9703 |
| resnet18_label_smooth_sgd_aug | 0.6640 | 1.2761 |
| resnet18_focal_rmsprop_no_aug | 0.6633 | 0.5754 |
| resnet18_ce_rmsprop_aug | 0.6405 | 1.0475 |
| resnet18_label_smooth_rmsprop_aug | 0.6228 | 1.3587 |
| resnet18_focal_rmsprop_aug | 0.6217 | 0.6101 |
| resnet18_focal_sgd_aug | 0.6197 | 0.6792 |
| vit_ce_adam_no_aug | 0.6013 | 1.0964 |
| vit_label_smooth_adam_no_aug | 0.5872 | 1.3873 |
| vit_ce_adam_aug | 0.5857 | 1.1583 |
| vit_focal_adam_no_aug | 0.5833 | 0.7061 |
| vit_label_smooth_adam_aug | 0.5572 | 1.4557 |
| vit_focal_adam_aug | 0.5401 | 0.7934 |
| vit_ce_sgd_no_aug | 0.4907 | 1.3936 |
| vit_label_smooth_sgd_no_aug | 0.4745 | 1.6236 |
| vit_focal_sgd_no_aug | 0.4667 | 0.9405 |
| vit_ce_sgd_aug | 0.4586 | 1.4809 |
| vit_focal_sgd_aug | 0.4435 | 1.0371 |
| vit_label_smooth_sgd_aug | 0.4322 | 1.7119 |
| vit_label_smooth_rmsprop_no_aug | 0.2830 | 1.9731 |
| vit_label_smooth_rmsprop_aug | 0.2075 | 2.1456 |
| vit_focal_rmsprop_aug | 0.1875 | 1.6709 |
| vit_focal_rmsprop_no_aug | 0.1693 | 1.6984 |
| vit_ce_rmsprop_no_aug | 0.1504 | 2.2541 |
| vit_ce_rmsprop_aug | 0.1075 | 2.2970 |

**Best Model:** `resnet18_ce_adam_no_aug` with an accuracy of **0.8019**.

---

## Analysis and Discussion

### 1. ViT vs ResNet-18
* **Performance:** The CNN baseline (ResNet-18) dominated the Vision Transformer (ViT) across all metrics. The highest performing ResNet-18 reached ~80.2% accuracy, while the best ViT configuration capped out at ~60.1%.
* **Inductive Bias vs. Flexibility:** Convolutional networks feature strong inductive biases, including locality and translation invariance. This makes them highly sample-efficient on small spatial datasets like CIFAR-10. ViTs process data as a sequence of patches and lack these innate spatial assumptions.
* **Data Regimes:** Because ViTs must "learn" the structural rules of images entirely from scratch using self-attention, they generally require massive datasets (like ImageNet-21k or JFT-300M) to match or outperform CNNs. On a small dataset like CIFAR-10, the CNN is naturally expected to perform better.

### 2. Effect of Data Augmentation
* **Generalization vs. Distortion:** Typically, horizontal and vertical flips are used to prevent overfitting and encourage generalizable feature learning. 
* **Observations:** Interestingly, configurations evaluated *without* augmentation (`no_aug`) generally outperformed their augmented counterparts. For example, `resnet18_ce_adam_no_aug` achieved 80.19% compared to `resnet18_ce_adam_aug` at 66.78%. 
* **Reasoning:** CIFAR-10 consists of highly low-resolution (32x32) images. Applying a *vertical* flip to objects with a strict natural orientation (e.g., cars, trucks, animals) creates unnatural instances that likely confuse the model and introduce detrimental noise rather than helpful variance.

### 3. Optimizer Comparison
* **Convergence and Stability:** **Adam** proved to be the vastly superior optimizer for both models. Its adaptive learning rate mechanics led to faster convergence and the highest overall accuracies.
* **SGD & RMSprop:** SGD requires meticulous hyperparameter tuning and generally settled at lower accuracies. RMSprop performed poorly across the board, but struggled catastrophically with the ViT architecture, producing the lowest accuracies in the entire experiment (10% - 28%).

### 4. Loss Function Comparison
* **Stability:** Standard **Cross-Entropy (CE)** yielded the most stable and performant model (`resnet18_ce_adam_no_aug`).
* **Focal Loss vs. Label Smoothing:** Focal loss, designed to penalize hard-to-classify examples, did not top the accuracy charts but did yield the lowest test loss across the board (`resnet18_focal_adam_no_aug` at 0.4074). This suggests Focal loss helped the model become highly confident in its correct predictions, even if the absolute number of correct classifications was slightly lower than with standard CE. Label Smoothing underperformed compared to CE, suggesting overconfidence was not the primary bottleneck for this task.
