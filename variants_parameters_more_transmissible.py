import numpy as np
import pandas as pd

from simulations_functions import make_k_matrix_from_M


# ------ parameters that require reading data

data_dir = './data'


# population size(N): borough_pop.pop_2020_estimate (see above)
# nyc population
borough_pop = pd.read_csv(data_dir+'/borough_population.csv')
# borough names
boroughs_order = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten']


# mobility data - average pairwise flows from FB data for good
# (all boroughs plus an other category)
# if unavailable, generate random data:
try:
	fb_mmatrix = pd.read_csv(data_dir+'/mmatrix_all_average_n_crisis.csv')
except FileNotFoundError:
	fb_mmatrix = pd.DataFrame(np.random.randint(low=10, high=100, size=(len(boroughs_order)+1, len(boroughs_order)+1)),
                           columns = boroughs_order+['other'],
                           index = boroughs_order+['other']).reset_index()
    

fb_mmatrix_scaled = fb_mmatrix.iloc[:,1:].values/fb_mmatrix.iloc[:,1:].sum(axis=1)[:,None]
# re-scale to the pop size
fb_mmatrix_scaled = fb_mmatrix_scaled[:-1, :-1] * borough_pop.pop_2020_estimate.values[:,None]

# construct the k-matrix
# avg contacts = 4
fb_kmatrix = make_k_matrix_from_M(fb_mmatrix_scaled, 4)


N = borough_pop.pop_2020_estimate



# ----- fixed parameter values / assumptions
p_q = 0.7  # probability of control measures given pos test
p_TP = 0.999  # true postive rate
p_FP = 1-p_TP
theta = 0.6  # infectiousness of individuals in quarantine reduced to theta*infectiousness
Q=5  # time in quarantine without symptoms


# ----- variant-specific parameters
# set-up for a delta-omicron BA.1 comparison
a_self = [0.6, 0.6]  # reduced susceptibility to reinfection (same variant)
a_cross = [0.5, 0.5]  # reduced susceptibility to reinfection (all other variants)
w_self = [1/90, 1/90]  # rate of waning full immunity (same variant) (assume full protection for 3 months / 1 month)
w_x = [1/60, 1/60]  # rate of waning full immunity (all other variants)
L = [3,3]  # latent period
D = [5,5]  # duration of infectiousness

b = [0.2,0.3]  # probability of infection given contact
    


