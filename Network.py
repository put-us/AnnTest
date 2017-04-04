
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
		for idx,_ in enumerate(dimentions[1:]):
			self.weights.append(np.zeros(shape=(dimentions[idx-1],dimentions[idx])))
			self.biases.append(np.zeros(shape=(dimentions[idx])))
	
	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))
		
	def sp(self,z):
		return self.sigmoid(z)*(1-self.sigmoid(z))
		
	def train(self,x,y):
		batch_len=len(x)/self.batch_size
		for i in range(self.epoch):
			
			for j in range(batch_len):
				self.SGD(x[j*batch_len:(j+1)*batch_len-1],y[j*batch_len:(j+1)*batch_len-1])
				
	
	def SGD(self,batch_in,batch_out):
		avg_del_C_ws=[np.zeros(shape=weight.shape)for weight in self.weights]
		avg_del_C_bs=[np.zeros(shape=bias.shape)for bias in self.biases]
		del_C_ws=[]
		del_C_bs=[]
		for input,output in zip(batch_in,batch_out):
			del_C_ws,del_C_bs=self.backProp(input,output)
			for i in range(len(avg_del_C_ws)):
				avg_del_C_ws[i]+=del_C_ws[i]
				avg_del_C_bs[i]+=del_C_bs[i]
		
		for i in range(len(avg_del_C_ws)):
			self.weights[i]=self.weights[i]-self.learning_rate/self.batch_size*avg_del_C_ws[i]
			self.biases[i]=self.biases[i]-self.learning_rate/self.batch_size*avg_del_C_bs[i]
		return True
		
	def backProp(self,input,output):
		weighted_sum=[]
		activation=[]
		zlist=[]
		del_C_ws=[np.zeros(shape=weight.shape)for weight in self.weights]
		del_C_bs=[np.zeros(shape=bias.shape)for bias in self.biases]
		for idx in range(len(self.weights)):
			z=self.weights[idx].dot(input)+self.biases[idx]
			z.append(z)
			activation.append(sigmoid(z))
			input=activation[idx]
		delta_L=(output-activation[-1])*self.sp(z[-1])
		del_C_ws[i]=acti*delta_L
		del_C_bs[i]=delta_L
		for i in range(-len(del_C_ws),0):

			delta_L=self.weights[idx].dot(delta_L)

		
			