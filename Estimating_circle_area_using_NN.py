import numpy as np 
import math
import scipy.constants as sc
import matplotlib.pyplot as plt

def circle_area(X):
	return sc.pi * X**2

# def relu(X):
	# return np.maximum(0, X)
	
# def Drelu(X):
	# return 1.0 * (X > 0)

def sigm(X):
	return 1.0 / (1.0 + np.exp(-X))

def Dsgm(X):
	return sigm(X) * (1.0 - sigm(X))

def softmax(X):
	return np.exp(X) / np.sum(np.exp(X), axis=1)

def plot_data(X,y,yp):
	plt.scatter(y, yp, color='blue', alpha=0.8) ## plot object 1
	Xmin, Xmax = plt.xlim()
	ymin, ymax = plt.ylim()
	#plt.plot([Xmin, Xmax], [ymin, ymax], color = 'red', lw=2, alpha= 0.8) ## plot object 2
	plt.scatter(y, yp, lw = 0.5, color='red', alpha=0.3) ## plot object 1
	#plt.plot(X, y, 'g^', X, y, 'bs')
	plt.xlabel('Actual')
	plt.ylabel('Predicted')
	plt.title('The circle area')
	#plt.savefig(file_name)
	plt.show()

## initial model with 1 hidden layer
def init_model(X, y, nodes):
	model = {}
	in_dim = X.shape[0]
	out_dim = y.shape[0]
	weight_1 = np.random.randn(nodes, in_dim)
	bias_1 = np.random.randn(nodes,1)
	weight_2 = np.random.randn(out_dim, nodes)
	bias_2 = np.random.randn(out_dim,1)
	model = {'W1':weight_1, 'b1':bias_1, 'W2':weight_2, 'b2':bias_2}
	return model

def foward_prog(X, model):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	z1 = np.dot(W1, X) + b1 ## logit
	s1 = sigm(z1) ## activation
	z2 = np.dot(W2, s1) + b2
	s2 = sigm(z2)
	weights = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}	
	out = {'z1': z1, 's1':s1, 'z2':z2, 's2':s2}
	return weights, out

## J(w,X) = 1 / m * sum(L(yp,ya))
## L(yp,y) = - (ylog(yp) + (1 - y)log(1 - yp))
def cal_loss(out, y):
	out = out['s2']
	# N = out.shape[0]
	# #loss = - (1.0 / N) * (np.dot(y.T, np.log(out))) + np.dot(1 - y.T, np.log(1 - out))
	# i = 0
	# loss = 0
	# dJ = 0
	# while i < N : 
		# error = out[i] - y[i]
		# loss += error**2
		# i+=1	
	error = out - y
	loss = np.mean(error**2)
	return loss

def backward_prog(weight, out, X, y):
	W1, b1, W2, b2 = weight['W1'], weight['b1'], weight['W2'], weight['b2']
	# i, N = 0, out['s2'].shape[0]
	# dJ_out = 0
	# ## at output layer
	# while i < N:
		# error = out['s2'][i] - y[i]
		# dJ_out += error * Dsgm(out['s2'][i])
		# i+=1
	# dJ_out = (1.0 / N) * dJ_out
	
	# ## at hidden layer
	# error = out['s2'] - y
	# delta_2 = error * Dsgm(out['s1'])	
	# dJW2 = np.dot(out['s1'].T, delta_2)
	#print(out['s1'].shape)
	#delta = np.dot()
	#dJW1 = 
	
	# at output layer
	error = out['s2'] - y
	Dsg_2 = Dsgm(out['z2'])
	#print(Dsg.shape, out['s1'].shape)
	delta_2 = np.dot((error * Dsg_2), out['s1'].T)
	dJW2 = {'W2':delta_2, 'b2':(error * Dsg_2)}
	
	#at hidden layer
	#print(W1.shape, out['s1'].shape, X.shape, W2.shape)
	Dsg_1 = Dsgm(out['z1'])
	delta_1 = np.dot(W2.T, error * Dsg_2) * Dsg_1
	dJW1 = {'W1':delta_1.dot(X.T), 'b1':delta_1}
	return dJW1, dJW2
	
def train():
	X = np.random.randn(356,1)
	X = np.abs(X)
	X_train = X
	y0 = circle_area(X_train)
	y = sigm(y0)
	model = init_model(X_train, y, nodes=3)
	#weight, out = foward_prog(X, y, model)
	#loss = cal_loss(out, y)
	#dJW1, dJW2 = backward_prog(weight, out, X, y)
	#print(out['s1'].shape, out['s2'].shape, y.shape)
	#print(loss)
	#plot_data(X,out['s2'])
	
	#training 
	done = False
	accepted = 0.01
	rate_learning = 0.01
	while done == False:
		weight, out = foward_prog(X_train, model)
		loss = cal_loss(out, y)
		if loss < accepted:
			print(loss)
			done = True
		else:
			dJW1, dJW2 = backward_prog(weight, out, X_train, y)
			weight['W1'] -= rate_learning * dJW1['W1']
			weight['b1'] -= rate_learning * dJW1['b1']
			weight['W2'] -= rate_learning * dJW2['W2']
			weight['b2'] -= rate_learning * dJW2['b2']
	yp = -np.log(1 / out['s2'] - 1)
	#print(yp, y0)
	plot_data(X_train,y0,yp)
	
	## predicted any values 
	# X_test = X[4:9,:]
	# model = {'W1':weight['W1'], 'b1':weight['b1'], 'W2':weight['W2'], 'b2':weight['b2']}
	# _, out = foward_prog(X_test, model)
	print('X_test=',X[111,],'predicted=',convert_to_actual(out['s2'][111]))

def convert_to_actual(X):
	return -np.log(1 / X - 1)

if __name__ == "__main__":
    train()
