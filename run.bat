@echo off
REM Setting Python Interpreter Path(your python path)
set python_path=C:\anaconda3\envs\FaithTrip\python.exe

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






