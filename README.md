# Plant Diseases Classification Projects

Plant disease can be recognised from the appearance of its infected leaves. For that reason, in this project, the images of leaves from bell pepper, tomato and potato will be used to identify whether the plant is health or infected. The challenges in this projects are:
1. The dataset is imbalance where one target only consists of 152 images meanwhile the other has 3202 images.
2. There are 3 type of plants where each plants has 3-10 disiease and healthy categories.

The source of dataset can be found [here](https://www.kaggle.com/datasets/arjuntejaswi/plant-village). The main challenges here is the imbalance dataset. One solution to handle the imbalance dataset is by implementing augmented technique to increase the number of dataset for the minority class. But in the train and validation set, there are only original images will be utilized. The augmented images will be applied only in train set. For that reason, the number of augmented images will depend on the number of original images available in train set. 

Projects with TensorFlow:
1. [Potato disease classification using partial dataset](https://github.com/imdwipayana/Plant_Disease_Project/blob/main/potatoes_disease_CNN.ipynb)
2. [Potato disease classification using all dataset](https://github.com/imdwipayana/Plant_Disease_Project/blob/main/potatoe_disease_CNN_full_dataset.ipynb)
3. [Potato disease classification using the augmented technique for minority dataset](https://github.com/imdwipayana/Plant_Disease_Project/blob/main/potato_disease_augmented.ipynb)
4. [Potato disease classification using pretrained model](https://github.com/imdwipayana/Plant_Disease_Project/blob/main/potato_disease_pre_trained_model.ipynb)
5. [Pepper, tomato and potato disease classification using all dataset](https://github.com/imdwipayana/Plant_Disease_Project/blob/main/all_diseases_all_dataset.ipynb)
6. [Pepper, tomato and potato disease classification using 152 dataset for each target](https://github.com/imdwipayana/Plant_Disease_Project/blob/main/all_disease_152_Images.ipynb)
7. [Pepper, tomato and potato disease classification using pre trained model](https://github.com/imdwipayana/Plant_Disease_Project/blob/main/all_plant_diseases_pretrained_model.ipynb)

Projects with PyTorch:
1. [Pepper, tomato and potato disease classification using pre trained model with PyTorch](https://github.com/imdwipayana/Plant_Disease_Project/blob/main/all_disease_classification_preTrained_PyTorch.ipynb)
2. [coming soon]()
