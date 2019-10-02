import math
import numpy as np
import matplotlib.pyplot as plt
## initializing a lattice
lattice = (150,150)

## coupling between spins
J = 1.0

## initializing an origin configuration 
def init_config():
	spins = [-1,1]
	return np.random.choice(spins, size=lattice)

## Here is ignore affecting of the external field
def cal_energy(config):
	sum_energy = (1.0 / 2.0) * np.sum(config * (
		np.roll(config, 1, axis=0) 
		+ np.roll(config, -1, axis=0)
		+ np.roll(config, 1, axis=1)
		+ np.roll(config, -1, axis=1)	
		))
	return -J * sum_energy

## Computing the energy for an configuration	
# def cal_energy(config):
	# sum_coupling = 0
	# N,M = lattice[0]-1, lattice[1]-1
	# i = 0
	# while i < N:
		# j = 0
		# while j < M:
			# sum_coupling = config[i,j] * (config[i-1,j] + config[i+1,j]+ config[i,j-1]+ config[i,j+1])
			# j+=1
	# i+=1		
	# return -J * sum_coupling	

## Calculate the probability
def cal_probability(X):
	return np.exp(X)

## calculate a new configuration
def cal_proposal_config(config):
	i = np.random.randint(config.shape[0])
	j = np.random.randint(config.shape[1])
	## flip spin at site (i,j) of current configuration lead to a new configuration
	new_config = np.copy(config) ## note use np.copy() to no relation with old array
	new_config[i,j] = -1 * new_config[i,j]
	return new_config
	
## run the Monte Carlo algorithm
def monte_carlo_alg(temp):
	N_conf = 50000
	current_config = init_config()
	#E = cal_energy(current_config)
	#Z = cal_probability(-E / temp)
	i = 0
	while i < N_conf:
	#while True:
		proposal_config = cal_proposal_config(current_config)		
		ratio = (1 / temp) * (cal_energy(current_config) - cal_energy(proposal_config))
		## Metropolis-Hastings Algorithm
		acceptance_probability = min(1,cal_probability(ratio))
		u = np.random.uniform(0, 1)			
		if acceptance_probability > u:
			#E += cal_energy(proposal_config) * cal_probability(-cal_energy(proposal_config)/ temp)
			#Z += cal_probability(-cal_energy(proposal_config)/ temp)		
			current_config = proposal_config			
		else:
			current_config = current_config
		
		# if i % 100 == 0:
			# plt.imshow(current_config, cmap='gray', vmin=-1, vmax=1, interpolation='none')
			# #im.set_data(spins)
			# #plt.draw()			
			# plt.pause(0.1)
		i+=1
		
	#E_avg = E / Z ## energy per site of the equilibrium configuration
	E_avg = cal_energy(current_config)
	M_avg = np.sum(current_config)
	return (E_avg / (lattice[0] * lattice[1]), M_avg / (lattice[0] * lattice[1]) ,current_config)

def plot_image(X, y, obj, path):
	plt.scatter(X, y, color='blue', alpha=0.8)
	plt.xlabel(obj['xlabel'])
	plt.ylabel(obj['ylabel'])
	plt.title(obj['title'])
	plt.savefig(path, dpi=300)
	plt.show()

def main():
	X = []
	y1 = [] 
	y2 = []
	min_temp = 1.0
	max_temp = 4.5
	delta = 0.1
	temp = min_temp
	# while temp < max_temp:	
		# print(temp)
		# E_site = monte_carlo_alg(temp)[0]
		# X.append(temp)
		# y1.append(E_site)
		
		# M_site = monte_carlo_alg(temp)[1]
		# y2.append(M_site)
		# temp += delta 
	while temp < max_temp:
		print(temp)
		plt.imshow(monte_carlo_alg(temp)[2], cmap='gray', vmin=-1, vmax=1, interpolation='none')		
		plt.pause(.01)
		temp += 1.0 
	
	## plot energy per site
	path1 = 'D:/Calculation/Image/energ150png.png'
	title1 = 'Energy per site of the equilibrium configuration versus temperature'	
	obj1 = {'xlabel': 'Temperature', 'ylabel':'Energy average', 'title': title1}
	plot_image(X, y1, obj1, path1)
	
	## plot magnetization per spin 
	path2 = 'D:/Calculation/Image/magne150png.png'
	title2 = 'magnetization per spin of the equilibrium configuration versus temperature'	
	obj2 = {'xlabel': 'Temperature', 'ylabel':'Magnetization', 'title': title2}
	plot_image(X, y2, obj2, path2)	
	
	## plot configuration 
	# a colormap and a normalization instance
	#cmap = plt.cm.jet
	#norm = plt.Normalize(vmin=-1, vmax=1)
	# map the normalized data to colors
	# image is now RGBA (512x512x4) 
	#data = monte_carlo_alg(1.5)[2]
	#image = cmap(norm(data))
	#plt.imsave('test.png', image)
	#plt.imshow(monte_carlo_alg(4.0)[2], cmap='gray', vmin=-1, vmax=1, interpolation='none')
	##temp = [1.0, 2.0, 4.0], size = [(30,30), (50,50), (100,100)]
	#monte_carlo_alg(1.5)[2]
	print('done')

if __name__=='__main__':
	main()