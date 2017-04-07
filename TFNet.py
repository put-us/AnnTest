import tensorflow as tf
import numpy as np
from mnist import MNIST

def label_to_vec(label_list):
	lable_vec=[]
	for label in  label_list:
		lable_vec.append([1 if x==label else 0 for x in range(10) ])
	return lable_vec
def list_to_tensor(list):
	newlist=[]
	for item in list :
		newlist.append(np.reshape(item,[1,len(item)]))
	tensor=np.reshape(newlist,[len(newlist),1,len(newlist[0])])
	return tensor
	
mndata = MNIST('MNIST')
images,labels=mndata.load_training()
#training_data=[(image,label) for image,label in  zip(images,labels)]#zip(image,labels)
image_tr,label_tr=mndata.load_testing()
lable_tr_list=label_to_vec(label_tr)
label_tr_vec=np.reshape(lable_tr_list,(len(lable_tr_list),len(lable_tr_list[0])))
lable_list=label_to_vec(labels)
label_vec=np.reshape(lable_list,(len(lable_list),len(lable_list[0])))

print("images",len(images), " " ,len(images[0]))
#print("lable_vec",len(lable_vec), "  ", len(lable_vec[0]))
image_tensor=np.reshape(images,(len(images),len(images[0])))
image_tr_tensor=np.reshape(image_tr,(len(image_tr),len(image_tr[0])))
print("image_tensor",image_tensor.shape)


batch=60000
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
x=tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

y=tf.nn.softmax(tf.matmul(x,W)+b)
cross_entropy=tf.reduce_mean(tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train=tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
r=int(len(images)/batch)
print("r", r,"  ")
for i in range(2000):
	#for image_batch,label_batch in zip(images[i*batch:(i+1)*batch-1],lable_list[i*batch:(i+1)*batch-1]):
		##CALL TENSORFLOW
		print("image_batch",len(images), " " )
		print()
		sess.run(train,feed_dict={x:images,y_:lable_list})
		
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy,feed_dict={x:image_tr_tensor,y_:label_tr_vec}))

