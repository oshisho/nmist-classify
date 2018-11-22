from keras.datasets import mnist
# from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.layers import Conv2D,MaxPool2D,BatchNormalization,Flatten
from keras.utils import np_utils
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# mnist=input_data.read_data_sets("./mnist",one_hot=True)
# x_train,y_train=mnist.train.images,mnist.train.labels
# x_test,y_test=mnist.test.images,mnist.test.labels
(x_train,y_train),(x_test,y_test)=mnist.load_data()
n_class=10
# print(x_train.shape)
# print(y_train.shape)

# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.imshow(x_train[i],cmap='gray',interpolation=None)
#     plt.title('class{}'.format(y_train[i]))
# plt.show()

x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train=x_train/255
x_test=x_test/255


y_train=np_utils.to_categorical(y_train,n_class)
y_test=np_utils.to_categorical(y_test,n_class)

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(28,28,1)))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))  # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))


# adam=Adam(lr=0.01)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,
          batch_size=128,epochs=4,
          verbose=0,
          validation_data=(x_test,y_test))

score=model.evaluate(x_test,y_test)
print('test_score:',score[0])
print('test_accuracy:',score[1])



