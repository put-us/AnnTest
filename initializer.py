import Network as an
from mnist import MNIST
mndata = MNIST('MNIST')
image,labels=mndata.load_training()
mndata.load_testing()
dimentions=[784,20,10]
logger_name=""
learning_rate=.1
batch_size=1
epoch=1
#print("label",labels[0])
ann=an.network(dimentions,logger_name,learning_rate,batch_size,epoch)

ann.train(image[:10],labels[:10])
print()
#print(ann.Test(x,y))