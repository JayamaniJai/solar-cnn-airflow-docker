

!pip install split_folders
!pip install tensorflow
!pip install keras
!pip install sklearn
!pip install seaborn
!pip install numpy
!pip install pandas
!pip install keras
!pip install matplotlib



#Some Basic Imports
import matplotlib.pyplot as plt #For Visualization
import numpy as np              #For handling arrays
import pandas as pd             # For handling data
import splitfolders
import os
import shutil
#Load solar cells label Data
solrDf = pd.read_csv('/content/drive/MyDrive/Solar-cell-CNN/solarcell-image-preprocessing-script-from-original-dataset/solar-cells-dataset/solarcells/solarcell-labels.csv')
solrDf['Image Path'] = solrDf['Image Path'].astype('str')
solrDf.dtypes

data_dir = '/content/drive/MyDrive/Solar-cell-CNN/solarcell-image-preprocessing-script-from-original-dataset/solar-cells-dataset'
severeDir = '/content/drive/MyDrive/Solar-cell-CNN/solarcell-image-preprocessing-script-from-original-dataset/solar-cells-dataset/solarcells/monocell/defective'
normalDir = '/content/drive/MyDrive/Solar-cell-CNN/solarcell-image-preprocessing-script-from-original-dataset/solar-cells-dataset/solarcells/monocell/functional'
os.makedirs(severeDir)
os.makedirs(normalDir)
for index, row in solrDf.iterrows():
        class_num = row['Defect Probability']
        solr_type = row['SolarCell Type']
        imagePath = os.path.join(data_dir,row['Image Path']).strip()
        if os.path.exists(imagePath.strip()):
            if (class_num >= 0.5) and (solr_type == 'mono'):
                shutil.copy(imagePath, severeDir)
            elif (class_num <= 0.5) and (solr_type == 'mono'):
                shutil.copy(imagePath, normalDir)
        else:
            print("File not exisit: ",imagePath)

#Spliting data into train, test and validation datasets
splitfolders.ratio('/content/drive/MyDrive/Solar-cell-CNN/solarcell-image-preprocessing-script-from-original-dataset/solar-cells-dataset/solarcells/monocell',output="/content/drive/MyDrive/Solar-cell-CNN/solarcell-image-preprocessing-script-from-original-dataset/solar-cells-dataset/DataforModel",
    seed=1337, ratio=(.7, .2, .1), group_prefix=None, move=False)

#Define Directories for train, test & Validation Set
train_path = '/content/drive/MyDrive/Solar-cell-CNN/solarcell-image-preprocessing-script-from-original-dataset/solar-cells-dataset/DataforModel/train'
test_path = '/content/drive/MyDrive/Solar-cell-CNN/solarcell-image-preprocessing-script-from-original-dataset/solar-cells-dataset/DataforModel/test'
valid_path = '/content/drive/MyDrive/Solar-cell-CNN/solarcell-image-preprocessing-script-from-original-dataset/solar-cells-dataset/DataforModel/val'
#Define some often used standard parameters
#The batch refers to the number of training examples utilized in one #iteration
batch_size = 32
#The dimension of the images we are going to define is 256 * 256 
img_height = 256
img_width = 256
print('done')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Create Image Data Generator for Train Set
image_gen = ImageDataGenerator(
                        rescale = 1./255,
                        horizontal_flip = True,
                        vertical_flip = True)
# Create Image Data Generator for Test/Validation Set
test_data_gen = ImageDataGenerator(rescale = 1./255,horizontal_flip = True,vertical_flip = True)
print('done')

train = image_gen.flow_from_directory(
      train_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary',
      batch_size=batch_size
      )

valid = test_data_gen.flow_from_directory(
      valid_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary', 
      batch_size=batch_size
      )

test = test_data_gen.flow_from_directory(
      test_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      shuffle=False, 
#setting shuffle as False just so we can later compare it with predicted values without having indexing problem 
      class_mode='binary',
      batch_size=batch_size
      )
print('done')

print(train.class_indices)

#Visualize the labeled data from train dataset
plt.figure(figsize=(12, 12))
for i in range(0, 10):
    plt.subplot(2, 5, i+1)
    for X_batch, Y_batch in train:
        image = X_batch[0]        
        dic = {0:'defective', 1:'functional'}
        plt.title(dic.get(Y_batch[0]))
        plt.axis('off')
        plt.imshow(np.squeeze(image),cmap='gray',interpolation='nearest')
        break
plt.tight_layout()
plt.show()

#Desining CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(activation = 'relu', units = 128))
cnn.add(Dense(activation = 'relu', units = 64))
cnn.add(Dense(activation = 'sigmoid', units = 1))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

plot_model(cnn,show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)
#cnn.summary()

#Value loss function
early = EarlyStopping(monitor='val_loss', mode='min', patience=5)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience = 5, 
                                            verbose=1,factor=0.3, 
                                            min_lr=0.000001)
callbacks_list = [ early, learning_rate_reduction]

#check the data balalnce between classes
dfiles = next(os.walk(severeDir))[2]
print("Number of total defective solar cells: ",len(dfiles))
ffiles = next(os.walk(normalDir))[2]
print("Number of total functional solar cells: ",len(ffiles))

#there is a imbalance in dataset, assigining class weight to penalize the misclassification made by the minority class 
#by setting a higher class weight and at the same time reducing weight for the majority class
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight(class_weight = "balanced",classes = np.unique(train.classes),y = train.classes)
cw = dict(zip(np.unique(train.classes), weights))
print(cw)

history = cnn.fit(train,epochs=25, validation_data=valid, class_weight=cw, callbacks=callbacks_list)

cnn.save('saved_model/my_model')

#pd.DataFrame(history.history).plot()

import matplotlib.pyplot as plt
epochs = [i for i in range(10)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

test_accu = cnn.evaluate(test)
print('The testing accuracy is :',test_accu[1]*100, '%')

preds = cnn.predict(test,verbose=1)

predictions = preds.copy()
#predictions
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns

cm = pd.DataFrame(data=confusion_matrix(test.classes, predictions, labels=[0, 1]),
                  index=["Actual Defective", "Actual Functional"],
columns=["Predicted Defective", "Predicted Functional"])

sns.heatmap(cm,annot=True,fmt="d")

print(classification_report(y_true=test.classes,
                            y_pred=predictions,
                            target_names =['defective','functional']))

test.reset()
x=np.concatenate([test.next()[0] for i in range(test.__len__())])
y=np.concatenate([test.next()[1] for i in range(test.__len__())])
print(x.shape)
print(y.shape)
#this little code above extracts the images from test Data iterator without shuffling the sequence
# x contains image array and y has labels 
dic = {0:'defective', 1:'functional'}
plt.figure(figsize=(20,20))
for i in range(0+100, 9+100):
    plt.subplot(3, 3, (i-100)+1)
    if preds[i, 0] >= 0.5: 
        out = ('{:.2%} probability of being functional case'.format(preds[i][0]))
    else: 
        out = ('{:.2%} probability of being defective case'.format(1-preds[i][0]))

    plt.title(out+"\n Actual case : "+ dic.get(y[i]))    
    plt.imshow(np.squeeze(x[i]))
    plt.axis('off')
    plt.show()

# Testing
defective_cell_path = '../input/mono-solar-cells-binary-dataset/mono-solarcells-binary/test/defective/cell0510.png'
#defective_cell_path = '../input/mono-solar-cells-binary-dataset/mono-solarcells-binary/test/functional/cell0312.png'
from tensorflow.keras.preprocessing import image
defect_img = image.load_img(defective_cell_path, target_size=(300, 300),color_mode='grayscale')
# Preprocessing the image
pp_img = image.img_to_array(defect_img)
pp_img = pp_img/255
pp_img = np.expand_dims(pp_img, axis=0)
#predict
preds= cnn.predict(pp_img)
#print
plt.figure(figsize=(6,6))
plt.axis('off')
if preds>= 0.5: 
    out = ('I am {:.2%} percent confirmed that this is a functional case'.format(preds[0][0]))
    
else: 
    out = ('I am {:.2%} percent confirmed that this is a defective case'.format(1-preds[0][0]))
plt.title("Solar cell\n"+out)  
plt.imshow(np.squeeze(pp_img))
plt.show()