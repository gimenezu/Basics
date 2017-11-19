#https://github.com/juliensimon/aws/blob/master/mxnet/intro/firstexample.py
import mxnet as mx
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

sample_count = 1000
train_count = 800
valid_count = sample_count - train_count

feature_count = 100
category_count = 10
batch=10

#create random X & Y
X = mx.nd.uniform(low=0, high=1, shape=(sample_count,feature_count))
Y = mx.nd.empty((sample_count,))
for i in range(0,sample_count-1):
  Y[i] = np.random.randint(0,category_count)

X_train = mx.nd.crop(X, begin=(0,0), end=(train_count,feature_count-1))
Y_train = Y[0:train_count]

X_valid = mx.nd.crop(X, begin=(train_count,0), end=(sample_count,feature_count-1))
Y_valid = Y[train_count:sample_count]

#print(X.shape, Y.shape, X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)

# Build network
data = mx.sym.Variable('data') #ici, on ne spécifie rien sur les données dans la couche d'entrée
fc1 = mx.sym.FullyConnected(data, name='fc1', num_hidden=64) #première couche cachée de 64 neurones, fully connected
relu1 = mx.sym.Activation(fc1, name='relu1', act_type="relu") #type d'activation de la couche.
fc2 = mx.sym.FullyConnected(relu1, name='fc2', num_hidden=category_count)#couche de sortie avec les catégories
out = mx.sym.SoftmaxOutput(fc2, name='softmax')
mod = mx.mod.Module(out)

# Build iterator
train_iter = mx.io.NDArrayIter(data=X_train,label=Y_train,batch_size=batch) #permet d'abstraire qu'il va récupérer 128 par 128 échantillons
#for batch in train_iter:
#  print batch.data
#  print batch.label

# Train model
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)#Fait le lien entr emodèle définit et les data.
mod.init_params(initializer=mx.init.Xavier(magnitude=2.))#quel initialisation des coeff des neurons.
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
mod.fit(train_iter, num_epoch=60)#nbre de fois ou le dataset passe dans le reseau.

#pred_iter = mx.io.NDArrayIter(data=X_train,label=Y_train, batch_size=batch)
#pred_count = train_count
pred_iter = mx.io.NDArrayIter(data=X_valid,label=Y_valid, batch_size=batch)#batch size : fait toute la backpropagation sur un échantillon ou sur un paquet /bcp plus rapide.
pred_count = valid_count

correct_preds = total_correct_preds = 0
print('batch [labels] [predicted labels]  correct predictions')
for preds, i_batch, batch in mod.iter_predict(pred_iter):
    label = batch.label[0].asnumpy().astype(int)
    pred_label = preds[0].asnumpy().argmax(axis=1)
    correct_preds = np.sum(pred_label==label)
    print i_batch, label, pred_label, correct_preds
    total_correct_preds = total_correct_preds + correct_preds

print('Validation accuracy: %2.2f' % (1.0*total_correct_preds/pred_count))

