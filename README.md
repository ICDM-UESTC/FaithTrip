# FaithTrip

## Brief Introduction

FaithTrip leverages Denoising Diffusion Probabilistic Models (DDPMs) to gradually align the generated trajectory with the tourist’s intent. The core idea stems from one of the characteristics of DDPM: the progressive data generation process. We employ an explicit condition-injecting strategy during the inference stage to achieve the alignment. This strategy progressively substitutes the source and destination of the generated trajectory with the ground truth of the source/destination (from the tourist’s query), enabling the model to iteratively refine itself and ultimately produce realistic, intent-consistent trajectories. 

## Detailed Showcase

This version has been submitted for review, and more information now is available at [here](https://joysmith99.github.io/FaithTrip-Showcase).

## Environmental Requirements

We run the code on a computer with RTX3060, i5 12400F, and 16G memory. Please Install the dependencies via anaconda:

### Create virtual environment

```
conda create -n FaithTrip python=3.9.18
```

### Activate environment

```
conda activate FaithTrip
```

### Install pytorch and cuda toolkit

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

### Install other requirements

```
conda install numpy pandas
pip install scikit-learn
```

## Folder Structure

| Folder Name |                         Description                          |
| :---------: | :----------------------------------------------------------: |
|    asset    |            Metadata and preprocessing operations             |
|    data     |                   Preprocessed input data                    |
|   results   |             Storage related experimental results             |
|    BASE     | Transformer-Based model(BASE) and the Base using clipping-merging strategy(BASE-CM) |
|  BASE-WSE   | Transformer-Based model using weighted classification loss in start and end points |
|  FaithTrip  |       the source code of FaithTrip with fast-sampling        |
| FaithTrip-Z |      FaithTrip predicts epsilon(EPS) and calculates x0       |
| FaithTrip-S |                 FaithTrip w/o fast-sampling                  |
|  README.md  |                  This instruction document                   |
|   run.bat   |              Script file for running this code               |

## How to run our programs

The detailed operation mode and parameter settings of each model can be found in **run.bat**. 

```
@echo off

REM Setting Python Interpreter Path
set python_path = Python location of your virtual environment

REM Run FaithTrip with Fast-Sampling
python .\FaithTrip\train_diffusion.py --dataset Osak --lr 0.01 --batch_size 4 --d_model 16 --step 32 --max_noise 0.02 --sampling 2

python .\FaithTrip\train_diffusion.py --dataset Glas --lr 0.01 --batch_size 4 --d_model 64 --step 32 --max_noise 0.02 --sampling 2

python .\FaithTrip\train_diffusion.py --dataset Edin --lr 0.01 --batch_size 16 --d_model 64 --step 32 --max_noise 0.06 --sampling 4

python .\FaithTrip\train_diffusion.py --dataset Toro --lr 0.01 --batch_size 8 --d_model 64 --step 32 --max_noise 0.06 --sampling 4

REM Run FaithTrip Standard w/o Fast-Sampling to predict x_0
REM The FaithTrip set default values T(step) in 32 and beta_t(max_noise) in 0.02, respectively. 
REM If you want to change it, try to add --step and --max_noise
python .\FaithTrip-S\train_diffusion.py --dataset Osak --lr 0.01 --batch_size 4 --d_model 16

python .\FaithTrip-S\train_diffusion.py --dataset Glas --lr 0.01 --batch_size 4 --d_model 64

python .\FaithTrip-S\train_diffusion.py --dataset Edin --lr 0.01 --batch_size 16 --d_model 64

python .\FaithTrip-S\train_diffusion.py --dataset Toro --lr 0.01 --batch_size 8 --d_model 64

REM Run FaithTrip-Z to predict noise
python .\FaithTrip-Z\train_diffusion.py --dataset Osak --lr 0.005 --batch_size 4 --d_model 32

python .\FaithTrip-Z\train_diffusion.py --dataset Glas --lr 0.005 --batch_size 4 --d_model 32

python .\FaithTrip-Ztrain_diffusion.py --dataset Edin --lr 0.005 --batch_size 16 --d_model 32

python .\FaithTrip-Z\train_diffusion.py --dataset Toro --lr 0.005 --batch_size 8 --d_model 32

REM Run BASE and BASE-CM
REM PS: max f1(pairs-f1) represent the results of BASE-CM and total f1(pairs-f1) represent the results of BASE
python .\BASE\train_base.py --dataset Osak --lr 0.001 --batch_size 4 --d_model 128

python .\BASE\train_base.py --dataset Glas --lr 0.001 --batch_size 4 --d_model 128

python .\BASE\train_base.py --dataset Edin --lr 0.001 --batch_size 16 --d_model 128

python .\BASE\train_base.py --dataset Toro --lr 0.001 --batch_size 8 --d_model 64

REM Run BASE-WSE
python .\BASE-WSE\train_base.py --dataset Osak --lr 0.001 --batch_size 4 --d_model 128 --se_weight 5

python .\BASE-WSE\train_base.py --dataset Glas --lr 0.001 --batch_size 4 --d_model 128 --se_weight 5

python .\BASE-WSE\train_base.py --dataset Edin --lr 0.001 --batch_size 16 --d_model 128 --se_weight 5

python .\BASE-WSE\train_base.py --dataset Toro --lr 0.001 --batch_size 8 --d_model 64 --se_weight 5
```

If your operating system is **Windows**, you can use the command In the working directory as

```
.\run.bat
```

to run this script file directly.  You can also directly paste commands into the terminal to run the program just like

```
python .\FaithTrip\train_diffusion.py --dataset Osak --lr 0.01 --batch_size 4 --d_model 16 --step 32 --max_noise 0.02 --sampling 2
```

Hope such an implementation could help you on your projects. Any comments and feedback are appreciated. If you find FaithTrip useful for your work please cite:

```
TODO
```
