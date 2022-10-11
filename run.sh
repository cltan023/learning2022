for seed in 1 2 3
do
    python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate 0.001 --batch_size_train 512 --seed ${seed}

    python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate 0.1 --batch_size_train 32 --seed ${seed}

    python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate 0.001 --batch_size_train 512 --seed ${seed}

    python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate 0.1 --batch_size_train 32 --seed ${seed}

    python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate 0.00001 --batch_size_train 512 --seed ${seed}

    python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate 0.001 --batch_size_train 32 --seed ${seed}
done

# use noisy label

for seed in 1 2 3
do
    for noise_y in 0.1 0.3 0.5 0.7 0.9
    do
        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate 0.001 --batch_size_train 512 --gpu_id 1 --noise_y ${noise_y} --seed ${seed} --save_dir 'noise_y_results'

        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate 0.1 --batch_size_train 32 --gpu_id 1 --noise_y ${noise_y} --seed ${seed} --save_dir 'noise_y_results'

        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate 0.001 --batch_size_train 512 --gpu_id 1 --noise_y ${noise_y} --seed ${seed} --save_dir 'noise_y_results'
        
        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate 0.1 --batch_size_train 32 --gpu_id 1 --noise_y ${noise_y} --seed ${seed} --save_dir 'noise_y_results'
        
        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate 0.00001 --batch_size_train 512 --gpu_id 1 --noise_y ${noise_y} --seed ${seed} --save_dir 'noise_y_results'
        
        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate 0.001 --batch_size_train 32 --gpu_id 1 --noise_y ${noise_y} --seed ${seed} --save_dir 'noise_y_results'
    done
done

for seed in 1 2 3
do
    for noise_x in 0.1 0.3 0.5 0.7 0.9
    do
        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate 0.001 --batch_size_train 512 --gpu_id 1 --noise_x ${noise_x} --seed ${seed} --save_dir 'noise_x_results'

        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate 0.1 --batch_size_train 32 --gpu_id 1 --noise_x ${noise_x} --seed ${seed} --save_dir 'noise_x_results'

        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate 0.001 --batch_size_train 512 --gpu_id 1 --noise_x ${noise_x} --seed ${seed} --save_dir 'noise_x_results'
        
        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate 0.1 --batch_size_train 32 --gpu_id 1 --noise_x ${noise_x} --seed ${seed} --save_dir 'noise_x_results'
        
        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate 0.00001 --batch_size_train 512 --gpu_id 1 --noise_x ${noise_x} --seed ${seed} --save_dir 'noise_x_results'
        
        python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate 0.001 --batch_size_train 32 --gpu_id 1 --noise_x ${noise_x} --seed ${seed} --save_dir 'noise_x_results'
    done
done