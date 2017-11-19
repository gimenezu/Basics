import mxnet as mx

######  1 ND arrays  ########
# mxnet deals with ndarrays as numpy. Supplémentary argument context must be set : cpu or gpu

gpu_device=mx.cpu() # Change this to mx.cpu() in absence of GPUs.

a = mx.nd.ones((100,100), mx.cpu()) # mx.gpu()
b = mx.nd.array([1,2,3], dtype=np.int32)

a.copyto(c)  # copy from CPU to GPU
e = b.as_in_context(c.context) + c  # same to above



#load & save

a = mx.nd.ones((2,3))
b = mx.nd.ones((5,6))
mx.nd.save("temp.ndarray", [a,b])
c = mx.nd.load("temp.ndarray")


#######   2 computation in backend engine ############"
import time
def do(x, n):
    """push computation into the backend engine"""
    return [mx.nd.dot(x,x) for i in range(n)]
def wait(x):
    """wait until all results are available"""
    for y in x:
        y.wait_to_read()

tic = time.time()
a = mx.nd.ones((1000,1000))
b = do(a, 50)
print('time for all computations are pushed into the backend engine:\n %f sec' % (time.time() - tic))
print(b)
wait(b)
print(b)
print('time for all computations are finished:\n %f sec' % (time.time() - tic))



######  3_ Symbiose API ######
#todo see to go further in model construction
# https://mxnet.incubator.apache.org/tutorials/basic/symbol.html
import mxnet as mx
a = mx.sym.Variable('a')
b = mx.sym.Variable('b')
c = a + b

#creation of  a NN
net = mx.sym.Variable('data')
# operation on this NN
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=128) #add one layer fully connected
net = mx.sym.Activation(data=net, name='relu1', act_type="relu") #activate it
net = mx.sym.FullyConnected(data=net, name='fc2', num_hidden=10) #add another layer
net = mx.sym.SoftmaxOutput(data=net, name='out')# todo ??
mx.viz.plot_network(net, shape={'data':(100,200)}) #todo how we can see ?
arg_name = net.list_arguments()  # get the names of the inputs
out_name = net.list_outputs()    # get the names of the outputs

#if we specify the weight
net = mx.symbol.Variable('data')
w = mx.symbol.Variable('myweight')
net = mx.symbol.FullyConnected(data=net, weight=w, name='fc1', num_hidden=128)

# infers output shape given the shape of input arguments
arg_shape, out_shape, _ = net.infer_shape(data=(2,3))
# infers output type given the type of input arguments
arg_type, out_type, _ = net.infer_type(data='float16')

# bind data & get results
ex = c.bind(ctx=mx.cpu(), args={'a' : mx.nd.ones([2,3]),
                                'b' : mx.nd.ones([2,3])})
ex.forward() # or eval (for bind & forward)
print('number of outputs = %d\nthe first output = \n%s' % (
           len(ex.outputs), ex.outputs[0].asnumpy()))

# save symbiose NN
print(c.tojson())
c.save('symbol-c.json')
c2 = mx.sym.load('symbol-c.json')


########### 4 _ Module
#once we have a NN, we have to forward / backward / evaluate it..
mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

#todo next lines
mod.fit(train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=8)

y = mod.predict(val_iter)
assert y.shape == (4000, 26)

#todo load & save grom a given iteration number
#https://mxnet.incubator.apache.org/tutorials/basic/module.html