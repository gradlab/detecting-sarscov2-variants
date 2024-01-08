################################################################################################### 
# prep data for viz & analysis: combine sim results - extreme strategies
################################################################################################### 



import numpy as np
import os



# set location of individual files
out_folder_preamble = 'extreme_strategies_intermediates/'


# set the dimensions of the individual data files, based on len of parameter values
extreme_proportions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
intro_times = [0, 50, 80, 100, 120, 150]
intro_locations = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten']
n_extprop = len(extreme_proportions)
n_extloc = len(intro_locations) # the location where tests are concentrated
n_variants = 2
n_iter = 100


# set the outputs
e_cumulative_all = np.zeros(shape=(n_extprop, n_extloc, len(intro_locations), len(intro_times),
                                   n_variants, len(intro_locations), n_iter) 
                            )
t_first_sequenced_all = np.zeros(shape=(n_extprop, n_extloc, len(intro_locations), len(intro_times),
                                   n_variants, n_iter) 
                            )


for ee, extreme_prop in enumerate(extreme_proportions): 
    for i, extreme_loc in enumerate(intro_locations): 
        out_folder = out_folder_preamble+'results_extreme-prop-'+str(ee)+'_extreme-loc-'+str(i)+'/'
        
        # time to detection
        # dimensions = 4
        # shape= (intro_locations x intro_times x n_variants x simulation iterations)
        t_first_sequenced_temp = np.load(out_folder+'t_first_sequenced_all.npy')
                
        # detection locations
        # dimensions = 5
        # shape = (intro_locations x intro_times x n_variants x n_locations x simulation iterations)
        patch_first_sequenced_temp = np.load(out_folder+'patch_first_sequenced_all.npy')
                

        # cumulative exposures, re-exposures as time series
        # dimensions = 6
        # shape = (intro_locations x intro_times x n_variants x n_locations x time x simulation iterations)
        e_cumulative_temp_ts = np.load(out_folder+'e_cumulative_all_ts.npy')
        e_re_cumulative_temp_ts = np.load(out_folder+'e_re_cumulative_all_ts.npy')
        
        e_cumulative_temp = np.zeros(shape=patch_first_sequenced_temp.shape)
        e_re_cumulative_temp = np.zeros(shape=patch_first_sequenced_temp.shape)
        # select slice @ time first sequenced
        for a in np.arange(t_first_sequenced_temp.shape[0]):
            for b in np.arange(t_first_sequenced_temp.shape[1]):
                for c in np.arange(t_first_sequenced_temp.shape[2]):
                    for d in np.arange(t_first_sequenced_temp.shape[3]):
                        the_time = t_first_sequenced_temp[a,b,c,d]
                        # adjust for the fact that t is zero when not sequenced
                        if the_time==0:
                            the_time = e_cumulative_temp_ts.shape[4]-1
                            print(the_time)
                        e_cumulative_temp[a,b,c,:,d] = e_cumulative_temp_ts[a,b,c,:,int(the_time),d]
                        e_re_cumulative_temp[a,b,c,:,d] = e_re_cumulative_temp_ts[a,b,c,:,int(the_time),d]
    
        e_temp = e_cumulative_temp+e_re_cumulative_temp
        
        
        e_cumulative_all[ee,i,:,:,:,:,:] = e_temp
        t_first_sequenced_all[ee,i,:,:,:,:] = t_first_sequenced_temp
        

# save
np.save(out_folder_preamble+'combo_e_cumulative_all.npy', e_cumulative_all)
np.save(out_folder_preamble+'combo_t_first_sequenced_all.npy', t_first_sequenced_all)






