rem --------------------------vgg_16_bn--------------------------------

rem @echo off
rem set root=C:\Users\huxf\Desktop\dyztmp\
rem set pojname=CAF
rem set arch=vgg_16_bn  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & %root%dyzenv\Scripts\python.exe main.py ^
rem --arch vgg_16_bn ^
rem --resume ../pre_train_model/CIFAR-10/vgg_16_bn.pt ^
rem --compress_rate [0.35]*5+[0.45]*3+[0.8]*5 ^
rem --num_workers 1 ^
rem --epochs 1 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 2 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir G:\data ^
rem & pause"

rem @echo off
rem set root=C:\Users\huxf\Desktop\dyztmp\
rem set pojname=CAF
rem set arch=vgg_16_bn  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & %root%dyzenv\Scripts\python.exe main.py ^
rem --arch vgg_16_bn ^
rem --from_scratch True ^
rem --resume final_pruned_model/vgg_16_bn_0.pt ^
rem --num_workers 1 ^
rem --job_dir %root%%pojname% ^
rem --epochs 200 ^
rem --lr 0.01 ^
rem --lr_decay_step 100,150 ^
rem --weight_decay 0.0005 ^
rem --dataset CIFAR10 ^
rem --data_dir G:\data ^
rem --save_id 1 ^
rem & pause"


rem --------------------------resnet_56--------------------------------

rem @echo off
rem set root=C:\Users\huxf\Desktop\dyztmp\
rem set pojname=CAF
rem set arch=resnet_56  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & %root%dyzenv\Scripts\python.exe main.py ^
rem --arch resnet_56 ^
rem --resume ../pre_train_model/CIFAR-10/resnet_56.pt ^
rem --compress_rate [0.]+[0.,0.]*1+[0.5,0.]*8+[0.,0.]*1+[0.5,0.]*8+[0.,0.]*1+[0.2,0.]*8 ^
rem --num_workers 1 ^
rem --epochs 1 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 1 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir G:\data ^
rem & pause"

rem @echo off
rem set root=C:\Users\huxf\Desktop\dyztmp\
rem set pojname=CAF
rem set arch=resnet_56  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & %root%dyzenv\Scripts\python.exe main.py ^
rem --arch resnet_56 ^
rem --from_scratch True ^
rem --resume final_pruned_model/resnet_56_0107_0.pt ^
rem --num_workers 1 ^
rem --job_dir %root%%pojname% ^
rem --epochs 400 ^
rem --lr 0.01 ^
rem --gpu 0 ^
rem --lr_decay_step 200,300 ^
rem --weight_decay 0.0005 ^
rem --dataset CIFAR10 ^
rem --data_dir G:\data ^
rem --save_id 1 ^
rem & pause"

rem --------------------------googlenet--------------------------------

rem @echo off
rem set root=C:\Users\huxf\Desktop\dyztmp\
rem set pojname=CAF
rem set arch=googlenet  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & %root%dyzenv\Scripts\python.exe main.py ^
rem --arch googlenet ^
rem --resume ../pre_train_model/CIFAR-10/googlenet.pt ^
rem --compress_rate [0.2]+[0.9]*24+[0.,0.3,0.] ^
rem --num_workers 4 ^
rem --epochs 1 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 2 ^
rem --weight_decay 0. ^
rem --dataset CIFAR10 ^
rem --data_dir G:\data ^
rem & pause"


rem @echo off
rem set root=C:\Users\huxf\Desktop\dyztmp\
rem set pojname=CAF
rem set arch=googlenet  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & %root%dyzenv\Scripts\python.exe main.py ^
rem --arch googlenet ^
rem --from_scratch True ^
rem --resume final_pruned_model/googlenet_1231_0.pt ^
rem --num_workers 1 ^
rem --job_dir %root%%pojname% ^
rem --epochs 200 ^
rem --lr 0.01 ^
rem --gpu 0 ^
rem --lr_decay_step 100,250 ^
rem --weight_decay 0.0005 ^
rem --dataset CIFAR10 ^
rem --data_dir G:\data ^
rem --save_id 1 ^
rem & pause"

rem --------------------------resnet_50--------------------------------

rem rem @echo off
rem set root=C:\Users\huxf\Desktop\dyztmp\
rem set pojname=CAF
rem set arch=resnet_50  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & %root%dyzenv\Scripts\python.exe main.py ^
rem --arch resnet_50 ^
rem --resume ../pre_train_model/ImageNet/resnet50.pth ^
rem --compress_rate [0.]+[0.2,0.2,0.2]*1+[0.75,0.75,0.2]*2+[0.2,0.2,0.2]*1+[0.75,0.75,0.2]*3+[0.2,0.2,0.2]*1+[0.75,0.75,0.2]*5+[0.2,0.2,0.2]+[0.2,0.2,0.2]*2 ^
rem --num_workers 4 ^
rem --epochs 1 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 2 ^
rem --batch_size 64 ^
rem --weight_decay 0. ^
rem --input_size 224 ^
rem --dataset ImageNet ^
rem --data_dir G:\ImageNet ^
rem & pause"

@echo off
set root=C:\Users\huxf\Desktop\dyztmp\
set pojname=CAF
set arch=resnet_50  
start cmd /c ^
"cd /D %root%%pojname%  ^
& %root%dyzenv\Scripts\python.exe main.py ^
--arch resnet_50 ^
--from_scratch True ^
--resume final_pruned_model/resnet_50_0202_0_7012_150.pt ^
--num_workers 4 ^
--job_dir %root%%pojname% ^
--epochs 120 ^
--batch_size 64 ^
--input_size 224 ^
--lr 0.01 ^
--lr_decay_step 30,60,90 ^
--weight_decay 0.0001 ^
--dataset ImageNet ^
--data_dir G:\ImageNet ^
--save_id 1 ^
& pause"


rem rem --------------------------mobilenet_v2--------------------------------

rem @echo off
rem set root=C:\Users\huxf\Desktop\dyztmp\
rem set pojname=CAF
rem set arch=mobilenet_v2  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & %root%dyzenv\Scripts\python.exe main.py ^
rem --arch mobilenet_v2 ^
rem --resume ../pre_train_model/ImageNet/mobilenet_v2.pth ^
rem --compress_rate [0.]+[0.1]*2+[0.1]*2+[0.3]+[0.1]*2+[0.3]*2+[0.1]*2+[0.3]*3+[0.1]*2+[0.3]*2+[0.1]*2+[0.3]*2+[0.1]*2 ^
rem --num_workers 2 ^
rem --epochs 1 ^
rem --job_dir %root%%pojname% ^
rem --lr 0.001 ^
rem --lr_decay_step 2 ^
rem --batch_size 2 ^
rem --weight_decay 0. ^
rem --input_size 224 ^
rem --dataset ImageNet ^
rem --data_dir G:\ImageNet ^
rem & pause"

rem @echo off
rem set root=C:\Users\huxf\Desktop\dyztmp\
rem set pojname=CAF
rem set arch=mobilenet_v2  
rem start cmd /c ^
rem "cd /D %root%%pojname%  ^
rem & %root%dyzenv\Scripts\python.exe main.py ^
rem --arch mobilenet_v2 ^
rem --from_scratch True ^
rem --resume final_pruned_model/mobilenet_v2_0.pt ^
rem --num_workers 1 ^
rem --job_dir %root%%pojname% ^
rem --epochs 150 ^
rem --batch_size 2 ^
rem --input_size 224 ^
rem --lr 0.01 ^
rem --lr_decay_step cos ^
rem --weight_decay 0.00005 ^
rem --bn_weight_decay 0.00005 ^
rem --dataset ImageNet ^
rem --data_dir G:\ImageNet ^
rem --save_id 1 ^
rem --warmup 5 ^
rem & pause"
