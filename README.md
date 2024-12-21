This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model achieves 99.44% accuracy in 20 epochs while maintaining parameter less than 20,000. Below is the architecture of network.

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
              ReLU-2            [-1, 8, 28, 28]               0
            Conv2d-3            [-1, 8, 28, 28]             584
              ReLU-4            [-1, 8, 28, 28]               0
       BatchNorm2d-5            [-1, 8, 28, 28]              16
            Conv2d-6            [-1, 8, 28, 28]             584
              ReLU-7            [-1, 8, 28, 28]               0
       BatchNorm2d-8            [-1, 8, 28, 28]              16
         MaxPool2d-9            [-1, 8, 14, 14]               0
           Conv2d-10           [-1, 16, 14, 14]           1,168
             ReLU-11           [-1, 16, 14, 14]               0
           Conv2d-12           [-1, 16, 14, 14]           2,320
             ReLU-13           [-1, 16, 14, 14]               0
      BatchNorm2d-14           [-1, 16, 14, 14]              32
        MaxPool2d-15             [-1, 16, 7, 7]               0
           Conv2d-16             [-1, 20, 5, 5]           2,900
             ReLU-17             [-1, 20, 5, 5]               0
           Conv2d-18             [-1, 10, 3, 3]           1,810
        AvgPool2d-19             [-1, 10, 1, 1]               0
================================================================
Total params: 9,510
Trainable params: 9,510
Non-trainable params: 0

Logs of test accuracy:
![11](https://github.com/user-attachments/assets/48db219c-f5de-4319-baf7-2c885a03b262)
![12](https://github.com/user-attachments/assets/c5e4ce10-a16b-4dca-b546-8469c741e9a9)
![13](https://github.com/user-attachments/assets/ed0125a9-adab-4b2d-9246-4c5e78f01846)

Test cases:
The Git hub actions file tests for below conditions:
1) The model has less than 20,000 parameters
2) The model has use batch normalization
3) The model has used Global Average Pool as last layer



