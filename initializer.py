import Network as an
from mnist import MNIST
mndata = MNIST('MNIST')
image,labels=mndata.load_training()
image_tr,label_tr=mndata.load_testing()
dimentions=[784,30,10]
logger_name=""
learning_rate=3
batch_size=10
epoch=30
#print("label",labels[0])
ann=an.network(dimentions,logger_name,learning_rate,batch_size,epoch)

ann.train(image[:],labels[:])
ann.printWB()
ann.Test(image_tr,label_tr)
print()
#print(ann.Test(x,y))
