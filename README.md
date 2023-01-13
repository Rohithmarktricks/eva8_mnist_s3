# EVA8 - 3 Backpropagation and Architectural Basics

## Assignment - Part 1
### Problem Statement
1. Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github. 
2. Use exactly the same values for all variables as used in the class
3. Take a screenshot, and show that screenshot in the readme file
4. The Excel file must be there for us to cross-check the image shown on readme (no image = no score)
5. Explain each major step
6. Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 

__please refer to the ```network.png``` image from the ```/img``` folder__
The Simple Neural Network is as follows.
![alt text](/imgs/network.png) 
1. It consists of:
    - 1. 1 Input Layer
    - 2. 1 Hidden Layer
    - 3. 1 Output Layer



### Solution
1. Please Refer to the ```Assignment1.xlsx``` spreadsheet from ```s3``` folder for the solution. 
2. However, the images/screenshots have been attached.

![alt text](/imgs/lr_0.1.png): The values computed for learning rate = 0.1

![alt text](/imgs/lr_0.2.png): The values computed for learning rate = 0.2

![alt text](/imgs/lr_0.5.png): The values computed for learning rate = 0.5

![alt text](/imgs/lr_0.8.png): The values computed for learning rate = 0.8

![alt text](/imgs/lr_1.0.png): The values computed for learning rate = 1.0

![alt text](/imgs/lr_2.0.png): The values computed for learning rate = 2.0


3. Final comparison of loss values (at different epochs) for different learning rates.
![alt text](/imgs/backprop_graph.png)



## Assignment - Part 2

To refactor the existing MNIST digit classification code such that it achieves:

1. 99.4% validation accuracy
2. Less than 20k Parameters
3. You can use anything from above you want. 
4. Less than 20 Epochs
5. Have used BN, Dropout, a Fully connected layer, have used GAP. 
6. To learn how to add different things we covered in this session, you can refer to this code: https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99 DONT COPY ARCHITECTURE, JUST LEARN HOW TO INTEGRATE THINGS LIKE DROPOUT, BATCHNORM, ETC.

#### Inputs
1. MNIST Dataset (Contains the Images and the corresponding labels)

#### Outputs
1. Labels of the corresponding MNIST images for training the Image classifier network.

### NN Architecture:
The Net2 architecture is used to solve the assignment. Contains the following layers:

    - 7 convolutional layers

    - 2 MaxPooling layers

    - 3 Linear/Fully connected layers

    ```Inputs:```
        __Image__ : 1x28x28 (MNIST Image)

        __RandomNumber__: 0-9
    
    ```Outputs:```
        __label__: Label of the MNIST Image

        __sum_output__: Sum of the predicted label of the MNIST Image and the random number.

### Loss Functions:
##### 1. Loss function of the MNIST image classification:
- Since it is a classification problem, ```nn.CrossEntropyLoss()``` API has been used to compute the cross entropy loss.

    ```
    Loss at Epoch 0: 1.3994579280100272
    Loss at Epoch 1: 0.2154729304313439
    Loss at Epoch 2: 0.20129536288048638
    Loss at Epoch 3: 0.027468014342368747
    Loss at Epoch 4: 0.007723255454994563
    Loss at Epoch 5: 0.055382025443283224
    Loss at Epoch 6: 0.07041706720206276
    Loss at Epoch 7: 0.004435703824696713
    Loss at Epoch 8: 0.0015469093507352036
    Loss at Epoch 9: 0.0031506284465735073 
    ```

### Optimizer:
```Adam``` Optimizer with learning rate ```lr=0.0001``` has been used as optimizer.


