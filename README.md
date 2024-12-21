This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model achieves 99.44% accuracy in 20 epochs while maintaining parameter less than 20,000. Below is the architecture of network.

Logs of test accuracy:

![11](https://github.com/user-attachments/assets/48db219c-f5de-4319-baf7-2c885a03b262)
![12](https://github.com/user-attachments/assets/c5e4ce10-a16b-4dca-b546-8469c741e9a9)
![13](https://github.com/user-attachments/assets/ed0125a9-adab-4b2d-9246-4c5e78f01846)

Test cases:
The Git hub actions file tests for below conditions:
1) The model has less than 20,000 parameters
2) The model has use batch normalization
3) The model has used Global Average Pool as last layer



