import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import podnn_tensorflow
import tensorflow as tf
from tensorflow.keras import Model
tf.random.set_seed(4)
import utils


n_samples = 500
X, y = make_circles(noise=0.3, random_state=17, n_samples=n_samples,factor=0.2)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)



X_train = tf.convert_to_tensor(x_train)
y_train = tf.convert_to_tensor(y_train.reshape(-1,1))
X_test = tf.convert_to_tensor(x_test)
y_test = tf.convert_to_tensor(y_test.reshape(-1,1))


unit_model_1 = [
    tf.keras.layers.Dense(12,activation='elu'),
    tf.keras.layers.Dense(10),
]

unit_model_2 = [
    tf.keras.layers.Dense(4)
]

class podnnModel(Model):
    def __init__(self):
        super(podnnModel, self).__init__()
        pass

    def build(self,input_shape):
        self.InputLayer = podnn_tensorflow.InputLayer(n_models=8)
        self.ParallelLayer1 = podnn_tensorflow.ParallelLayer(unit_model_1)
        self.OrthogonalLayer = podnn_tensorflow.OrthogonalLayer1D()
        self.AggregationLayer = podnn_tensorflow.AggregationLayer(stride=2)
        self.DenseLayer = tf.keras.layers.Dense(1, activation='sigmoid',name='last_dense')

    def call(self,x):
        x = self.InputLayer(x)
        x = self.ParallelLayer1(x)
        x = self.OrthogonalLayer(x)
        x_orth = self.AggregationLayer(x)
        x = self.DenseLayer(x_orth)
        return x,x_orth



loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')


model = podnnModel()


@tf.function
def train_step(x, labels):

      with tf.GradientTape() as tape:
            predictions,_ = model(x)
            loss = loss_object(labels, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      train_loss(loss)
      train_accuracy(labels, tf.squeeze(predictions))


epochs = 200
for i in range(epochs):
      train_loss.reset_states()
      train_accuracy.reset_states()

      train_step(X_train, y_train)

      if np.mod(i,10)==0:
          print('epoch:'+str(i)+'  train loss='+str(train_loss.result()))
          print('epoch:'+str(i) + '   train accuracy=' + str(train_accuracy.result()))


preds_test,_ = model(X_test)
test_acc = accuracy_score(y_test,np.round(preds_test))
print('=======> test accuracy=' + str(test_acc))

utils.plot_bounday_tensorflow(model,4,x_train,y_train,x_test,y_test)