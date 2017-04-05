
# NETWORK 
import numpy as np
import logging 
class network:
	def __init__(self,dimentions,logger_name,learning_rate,batch_size,epoch):
		self.module_logger_name=logger_name+'.Network'
		self.logger=logging.getLogger(self.module_logger_name)
		self.logger.debug("Ann Dimesions : %s"%dimentions)
		self.weights=[]
		self.biases=[]
		self.learning_rate=learning_rate
		self.batch_size=batch_size
		self.epoch=epoch
		for idx in range(len(dimentions)-1):
                        #print(idx)
                        self.weights.append(np.zeros(shape=(dimentions[idx+1],dimentions[idx])))
                        self.biases.append(np.zeros(shape=(dimentions[idx+1],1)))
		#print(self.weights)
		#print(self.biases)
	
	def sigmoid(self,z):

                return 1.0/(1.0+np.exp(-z))
		
	def sp(self,z):
                #print("in SP Z",z)
                #print("in SP Prime",self.sigmoid(z)*(1-self.sigmoid(z)))
                return self.sigmoid(z)*(1-self.sigmoid(z))
		
	def train(self,x,y):
                print(len(x))
                batch_len=int(len(x)/self.batch_size)
                for i in range(self.epoch):
                        for j in range(batch_len):
                                self.SGD(x[j*batch_len:(j+1)*batch_len-1],y[j*batch_len:(j+1)*batch_len-1])
				
	
	def SGD(self,batch_in,batch_out):
		avg_del_C_ws=[np.zeros(shape=weight.shape)for weight in self.weights]
		avg_del_C_bs=[np.zeros(shape=bias.shape)for bias in self.biases]
		del_C_ws=[]
		del_C_bs=[]
		for input1,output in zip(batch_in,batch_out):
			del_C_ws,del_C_bs=self.backProp(input1,output)
			for i in range(len(avg_del_C_ws)):
                                print("AVg del C" ,i)
                                avg_del_C_ws[i]+=del_C_ws[i]
                                avg_del_C_bs[i]+=del_C_bs[i]
		
		for i in range(len(avg_del_C_ws)):
			self.weights[i]=self.weights[i]-self.learning_rate/self.batch_size*avg_del_C_ws[i]
			self.biases[i]=self.biases[i]-self.learning_rate/self.batch_size*avg_del_C_bs[i]
		return True
		
	def backProp(self,input1,output):
		weighted_sum=[]
		activation=[]
		zlist=[]
		del_C_ws=[np.zeros(shape=weight.shape)for weight in self.weights]
		del_C_bs=[np.zeros(shape=bias.shape)for bias in self.biases]
		for idx in range(len(self.weights)):
                        if idx==0:
                                z=np.dot(self.weights[idx],input1)+self.biases[idx].transpose()
                        else:
                                z=np.dot(self.weights[idx],input1)+self.biases[idx]
                        #print(idx, z.shape,z)
                        zlist.append(z.transpose())
                        activation.append(self.sigmoid(z.transpose()))
                        #print(idx, activation[idx].shape,activation[idx])
                        input1=activation[idx]
		#print("Output", zlist[-1])
		delta_L=(output-activation[-1])*self.sp(zlist[-1])
		del_C_ws[-1]=np.dot(activation[-2],delta_L)
		del_C_bs[-1]=delta_L
		for idx in range(-1,-len(del_C_ws)):
			print("idx2",idx)
			delta_L=self.weights[idx].dot(delta_L)*self.sp(zlist[idx-1])
			if idx==-len(del_C_ws)+1:
                                del_C_ws[idx]=(input1*delta_L).transpose()
                                print(del_C_ws[idx])
			else :
                                del_C_ws[idx]=(activation[idx-2]*delta_L).transpose()
                                print(del_C_ws[idx])
		
			del_C_bs[idx]=delta_L
		#print(del_C_ws,del_C_bs)
		
		return (del_C_ws,del_C_bs)
	
	def Test(self,input1):
		output=-1
		for idx in range(len(self.weights)):
			output=self.sigmoid(self.weights[idx].dot(input1)+self.biases[idx])
			#z.append(zs)
			#activation.append(sigmoid(z))
			input1=output
		return np.argmax(output)
