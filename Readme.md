# Pix2Flows

Computational Fluid Dynamics (CFD) is a resource intensive solution to fluid flow modelling. These pose a serious contraint on usability and accessibility of such methods. Data driven methods are a good approach for combating these issues but are limited in terms of reliability and physical correctness. This project aims to solve these two problems at the same time. I propose Pix2Flows, a Physics-informed Conditional Generative Adversarial Network (GAN) based on modified Pix2Pix architecture for prediction of flow fields around custom geometries. This is in extenstion to my work on [DeepSteadyFlows](https://github.com/ritvik03/DeepSteadyFlows) and produces better performance than it on test cases as well as custom geometries. It also gets rid of the visual artifact present in previous methods and is more robust to complex shape transformation.

![thumbnail](image_assets/thumbnail.png)

## Model Evolution
Below animation shows evolution of model results on one particular geometry between epoch 0 to 180.

![evolution](image_assets/evolution.gif)

![l1_loss](image_assets/l1_loss.png)

# Results
Few randomly picked results

![result](result_images/0.png)
***
![result](result_images/8.png)
***
![result](result_images/4.png)
***
![result](result_images/7.png)
***
![result](result_images/5.png)
***
![result](result_images/6.png)


# Custom Results
![result](result_images/1.png)
![result](result_images/2.png)
![result](result_images/3.png)

# Usage

> **$ python3 draw.py**
- Draw the shape keeping the mid point of canvas enclosed in the buff body drawn

- Press "m" to fillup

- Press "Esc" to save and terminate.

> **$ python3 pred.py**
- loads up model from checkpoint
- shows prediction as matplotlib figure
- saves prediction as custom.png

## Architecture
**[NOTE]** Add details about architecture later

### Generator
![generator](image_assets/generator_architecture.png)

![generator](image_assets/pix2flows_generative_model.png)

### Discriminator
![discriminator](image_assets/pix2flows_discriminator_model.png)


## Loss function

### Discriminator
Discriminator Loss function is sum of cross-entropy losses for determination of real and fake samples

### Generator

Generator Loss has 4 components:
> **Fake generation loss** : how good the generator is at fooling the discriminator

> **Mean absolute error** : How closely the predictions matches the true output

> **NS-loss** : How much the predictions deviate from real world physics (Navier-Stokes equation)

> **Gradient Loss** : Deviation in x and y spatial gradients of predicted fields from true fields

**NS-loss** : e<sub>x</sub><sup>2</sup> + e<sub>y</sub><sup>2</sup>

- <strong>e<sub>x</sub></strong> : Error in steady state navier-stokes equation in X-direction

- <strong>e<sub>y</sub></strong> : Error in steady state navier-stokes equation in Y-direction

- <strong>continuity_loss</strong> : Square of error in continuity equation in 2 dimensions

### Final Generator loss function :
**Loss = Fake generation loss + λ<sub>1</sub> MAE +  λ<sub>2</sub> NS-loss + λ<sub>3</sub> Gradient_loss**

- λ<sub>1</sub> = 100
- λ<sub>2</sub> = 10000
- λ<sub>3</sub> = 100

[Note]: More tinkering with network architecture and loss hyperparameters required. Also, generalize this to 3D

## About me
This repository is made by Ritvik Pandey for the academic use as better implementation of my Maters Thesis Project (MTP) for Department of Mechanical Engineering, Indian Institute of Technology Kharagpur (IIT-KGP)

Contact: ritvik.pandey03@gmail.com


