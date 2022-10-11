# for learning_rate in 0.4 0.2 0.1 0.05 0.01 0.01
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# for learning_rate in 0.2 0.1 0.05 0.01 0.001
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# for learning_rate in 0.01 0.005 0.001 0.0005 0.0001 0.00005
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate 0.001 --batch_size_train 512

# python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate 0.1 --batch_size_train 32

# python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate 0.00001 --batch_size_train 512

# python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate 0.001 --batch_size_train 32


# for learning_rate in 0.4 0.2 0.1 0.05 0.01
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'densenet121' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# for learning_rate in 0.2 0.1 0.05 0.01 0.001
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# for learning_rate in 0.01 0.005 0.001 0.0005 0.0001 0.00005
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'resnet18' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# for learning_rate in 0.05 0.03 0.01 0.005 0.001
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'vgg16_bn' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done


# for learning_rate in 0.4 0.2 0.1 0.05 0.01 0.01
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'resnet18' --data_name 'cifar100' --num_classes 100 --num_of_per_class 500 --optimizer 'SGD' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# for learning_rate in 0.2 0.1 0.05 0.01 0.001
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'resnet18' --data_name 'cifar100' --num_classes 100 --num_of_per_class 500 --optimizer 'SGDM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# for learning_rate in 0.01 0.005 0.001 0.0005 0.0001 0.00005
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'resnet18' --data_name 'cifar100' --num_classes 100 --num_of_per_class 500 --optimizer 'ADAM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# for learning_rate in 0.05 0.03 0.01 0.005 0.001
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'vgg16_bn' --data_name 'cifar100' --num_classes 100 --num_of_per_class 500 --optimizer 'SGD' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# for learning_rate in 0.4 0.2 0.1 0.05 0.01
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'densenet121' --data_name 'cifar100' --num_classes 100 --num_of_per_class 500 --optimizer 'SGD' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train}
#     done
# done

# for learning_rate in 0.01 0.008 0.006 0.004 0.001
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'vgg16_bn' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 1
#     done
# done

# for learning_rate in 0.01 0.008 0.006 0.004 0.001
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'vgg16_bn' --data_name 'cifar100' --num_classes 100 --num_of_per_class 500 --optimizer 'SGDM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 1
#     done
# done

# for learning_rate in 0.2 0.1 0.05 0.01 0.005
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'densenet121' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 1
#     done
# done

# for learning_rate in 0.2 0.1 0.05 0.01 0.005
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'densenet121' --data_name 'cifar100' --num_classes 100 --num_of_per_class 500 --optimizer 'SGDM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 1
#     done
# done


# for learning_rate in 0.001 0.0008 0.0001 0.00005
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'vgg16_bn' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 0
#     done
# done

# for learning_rate in 0.001 0.0008 0.0001 0.00005
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'vgg16_bn' --data_name 'cifar100' --num_classes 100 --num_of_per_class 500 --optimizer 'ADAM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 0
#     done
# done


# for learning_rate in 0.005 0.001 0.0005 0.0001 0.00005
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'densenet121' --data_name 'cifar10' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 0
#     done
# done

# for learning_rate in 0.005 0.001 0.0005 0.0001 0.00005
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'densenet121' --data_name 'cifar100' --num_classes 100 --num_of_per_class 500 --optimizer 'ADAM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 0
#     done
# done


# for learning_rate in 0.05 0.03 0.01 0.005 0.001
# do
#     for batch_size_train in 32 64 128 256 512
#     do
#         python main.py --arch 'vgg16_bn' --data_name 'svhn' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 1
#     done
# done

# for learning_rate in 0.01 0.008 0.006 0.004 0.001
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'vgg16_bn' --data_name 'svhn' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 1
#     done
# done

# for learning_rate in 0.001 0.0008 0.0005 0.0001 0.00005
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'vgg16_bn' --data_name 'svhn' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 1
#     done
# done

# for learning_rate in 0.4 0.2 0.1 0.05 0.01
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'densenet121' --data_name 'svhn' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGD' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 0
#     done
# done

# for learning_rate in 0.2 0.1 0.05 0.01 0.005
# do
#     for batch_size_train in 64 128 256 512 1024
#     do
#         python main.py --arch 'densenet121' --data_name 'svhn' --num_classes 10 --num_of_per_class 5000 --optimizer 'SGDM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 0
#     done
# done

# for learning_rate in 0.005 0.001 0.0005 0.0001 0.00005
# do
#     for batch_size_train in 128 256 512 1024
#     do
#         python main.py --arch 'densenet121' --data_name 'svhn' --num_classes 10 --num_of_per_class 5000 --optimizer 'ADAM' --learning_rate ${learning_rate} --batch_size_train ${batch_size_train} --gpu_id 0
#     done
# done