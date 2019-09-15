import numpy as np
import matplotlib.pyplot as plt
## initializing a lattice
lattice = (250,250)

## coupling between spins
J = 1.0

## initializing an origin configuration 
def init_config():
	spins = [-1,1]
	return np.random.choice(spins, size=lattice)

## Here is ignore affecting of the external field
def cal_energy(config):
	sum_energy = np.sum(config * (
		np.roll(config, 1, axis=0) 
		+ np.roll(config, -1, axis=0)
		+ np.roll(config, 1, axis=1)
		+ np.roll(config, -1, axis=1)	
		))
	return -J * sum_energy

## Calculate the probability
def cal_probability(ratio):
	return np.exp(ratio)

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
	N_conf = 10000
	current_config = init_config()
	#E = cal_energy(current_config)
	#E2 = E**2
	i = 0
	while i < N_conf:
		proposal_config = cal_proposal_config(current_config)		
		ratio = (1 / temp) * (cal_energy(current_config) - cal_energy(proposal_config))
		## Metropolis-Hastings Algorithm
		acceptance_probability = min(1,cal_probability(ratio))
		u = np.random.uniform(0, 1)		
		if acceptance_probability > u:
			current_config = proposal_config
			#E += cal_energy(proposal_config)
			#E2 += E**2
		else:
			current_config = current_config
		i+=1
	E_site = cal_energy(current_config) ## energy per site of the equilibrium configuration
	return (E_site / lattice[0]**2, current_config)

def plot_image(X, y):
	plt.scatter(X, y, color='blue', alpha=0.8)
	plt.xlabel("Temperature")
	plt.ylabel("Energy per site")
	plt.title("Energy per site of the equilibrium configuration versus temperature")
	plt.show()

def main():
	X = []
	y = [] 
	min_temp = 0.1
	max_temp = 5.0
	delta = 0.01
	temp = min_temp
	while temp < max_temp:	
		print(temp)
		E_site = monte_carlo_alg(temp)[0]
		X.append(temp)
		y.append(E_site)
		temp += delta 		
	plot_image(X, y)
	#plt.imshow(monte_carlo_alg(1.5)[1], cmap='gray', vmin=-1, vmax=1, interpolation='none')
	print('done')

if __name__=='__main__':
	main()