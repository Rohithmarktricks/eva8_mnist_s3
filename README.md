# EVA8 - 3 Backpropagation and Architectural Basics

## Assignment - Part 1
### Problem Statement
1. Rewrite the whole excel sheet showing backpropagation. Explain each major step, and write it on Github. 
2. Use exactly the same values for all variables as used in the class
3. Take a screenshot, and show that screenshot in the readme file
4. The Excel file must be there for us to cross-check the image shown on readme (no image = no score)
5. Explain each major step
6. Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 

### Inputs and Outputs:
Inputs and outputs of the simple NN shown above are as follows:
![alt text](/imgs/inputs_outputs.png)

#### Note: 
```
i1, i2 -> Inputs
t1, t2 -> Outputs
```



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
1. Please refer to the ```MNIST_BN_Dropout.ipynb``` Jupyter notebook from ```s3``` folder for the solution.
```
    This Model architecture is used to solve the assignment.
    Contains the following layers:
        2 convolutional layers
        1 MaxPooling layers
        2 Linear/Fully connected layers
        2 Batch Normalization layers
        1 Dropout Layer
        1 Global Average Pooling Layer
    Inputs:
        Image : 1x28x28 (MNIST Image)
    
    Outputs:
        label: Label of the MNIST Image
```

2. CNN Model - model summary: A Network with 19822 trainable parameters has been trained!
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 28, 28]             640
              ReLU-2           [-1, 64, 28, 28]               0
       BatchNorm2d-3           [-1, 64, 28, 28]             128
         MaxPool2d-4           [-1, 64, 14, 14]               0
            Conv2d-5           [-1, 32, 14, 14]          18,464
              ReLU-6           [-1, 32, 14, 14]               0
       BatchNorm2d-7           [-1, 32, 14, 14]              64
           Dropout-8                   [-1, 32]               0
            Linear-9                   [-1, 12]             396
           Linear-10                   [-1, 10]             130
================================================================
Total params: 19,822
Trainable params: 19,822
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.39
Params size (MB): 0.08
Estimated Total Size (MB): 1.47
----------------------------------------------------------------
```

### Loss Functions:
##### 1. Loss function of the MNIST image classification:
- Since it is a classification problem, ```nn.CrossEntropyLoss()``` API has been used to compute the cross entropy loss.

    ```
    loss=0.9017577171325684 batch_id=234: 100%|██████████████████████████████████████████| 235/235 [00:08<00:00, 26.63it/s]
    loss=0.38888445496559143 batch_id=234: 100%|█████████████████████████████████████████| 235/235 [00:08<00:00, 27.89it/s]
    loss=0.3208557367324829 batch_id=234: 100%|██████████████████████████████████████████| 235/235 [00:09<00:00, 25.13it/s]
    loss=0.1358419507741928 batch_id=234: 100%|██████████████████████████████████████████| 235/235 [00:08<00:00, 27.08it/s]
    loss=0.14185713231563568 batch_id=234: 100%|█████████████████████████████████████████| 235/235 [00:08<00:00, 27.05it/s]
    loss=0.2285839319229126 batch_id=234: 100%|██████████████████████████████████████████| 235/235 [00:08<00:00, 27.66it/s]
    loss=0.0762401819229126 batch_id=234: 100%|██████████████████████████████████████████| 235/235 [00:08<00:00, 27.57it/s]
    loss=0.058386173099279404 batch_id=234: 100%|████████████████████████████████████████| 235/235 [00:08<00:00, 27.47it/s]
    loss=0.09039608389139175 batch_id=234: 100%|█████████████████████████████████████████| 235/235 [00:08<00:00, 27.56it/s]
    loss=0.09699162095785141 batch_id=234: 100%|█████████████████████████████████████████| 235/235 [00:08<00:00, 26.92it/s]
    loss=0.08095577359199524 batch_id=234: 100%|█████████████████████████████████████████| 235/235 [00:08<00:00, 27.43it/s]
    loss=0.05563954636454582 batch_id=234: 100%|█████████████████████████████████████████| 235/235 [00:08<00:00, 26.60it/s]
    loss=0.052124977111816406 batch_id=234: 100%|████████████████████████████████████████| 235/235 [00:08<00:00, 27.46it/s]
    loss=0.05465966835618019 batch_id=234: 100%|█████████████████████████████████████████| 235/235 [00:08<00:00, 28.07it/s]
    loss=0.11855859309434891 batch_id=234: 100%|█████████████████████████████████████████| 235/235 [00:08<00:00, 28.15it/s]
    loss=0.015947774052619934 batch_id=234: 100%|████████████████████████████████████████| 235/235 [00:08<00:00, 27.85it/s]
    loss=0.060164760798215866 batch_id=234: 100%|████████████████████████████████████████| 235/235 [00:08<00:00, 28.14it/s]
    loss=0.05754023417830467 batch_id=234: 100%|█████████████████████████████████████████| 235/235 [00:08<00:00, 28.22it/s]
    loss=0.043878063559532166 batch_id=234: 100%|████████████████████████████████████████| 235/235 [00:08<00:00, 27.84it/s]
    loss=0.007101322989910841 batch_id=234: 100%|████████████████████████████████████████| 235/235 [00:08<00:00, 28.00it/s]
    ```
#### Testing the model: Loss and Accuracy values:
The above model on the testing dataset performed as below:
```Test set: Average loss: 0.0555, Accuracy: 9831/10000 (98%)```

### Optimizer:
```Adam``` Optimizer with learning rate ```lr=0.0001``` has been used as optimizer.


