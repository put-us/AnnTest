import Network as an
import ANetwork as ann
from mnist import MNIST
mndata = MNIST('MNIST')
images,labels=mndata.load_training()
training_data=[(image,label) for image,label in  zip(images,labels)]#zip(image,labels)
image_tr,label_tr=mndata.load_testing()
test_data=[(image,label) for image,label in  zip(image_tr,label_tr)]#zip(image_tr,label_tr)
dimentions=[784,30,10]
logger_name=""
learning_rate=3
batch_size=10
epoch=30
#print("label",labels[0])
#ann=an.network(dimentions,logger_name,learning_rate,batch_size,epoch)
net=ann.Network(dimentions)
net.SGD(training_data, epoch, batch_size, learning_rate, test_data=test_data)
a#nn.train(image[:],labels[:])
#ann.printWB()
#ann.Test(image_tr,label_tr)
print()
#print(ann.Test(x,y))
