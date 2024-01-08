
################################################################################################### 
# prep data for viz & analysis: combine sim results - fixed resources sensitivity analysis V2
################################################################################################### 



import numpy as np
import os




# set location of individual files
out_folder_preamble = 'fixed_resources/'


# set the dimensions of the individual data files, based on len of parameter values
quantity_tests = [500, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10500, 12500, 14500, 16500]
quantity_sequences = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 
                      3000, 3500, 4500, 5500, 6500, 7500, 9000, 10500, 12000, 13500, 15000]                
intro_locations = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten']
intro_times = [0, 50, 80, 100, 120, 150]
n_multipliers = len(quantity_tests)
n_seqrates = len(quantity_sequences)
n_variants = 2
n_iter = 100



# set the outputs
e_cumulative_all = np.zeros(shape=(n_multipliers, 
                                   n_seqrates, len(intro_locations), len(intro_times),
                                   n_variants, len(intro_locations), n_iter) 
                            )

t_first_sequenced_all = np.zeros(shape=(n_multipliers, 
                                   n_seqrates, len(intro_locations), len(intro_times),
                                   n_variants, n_iter) 
                            )


for seq_iter in np.arange(len(quantity_sequences)):
    seq_quantity = quantity_sequences[seq_iter]
    
    for m in np.arange(len(quantity_tests)):
        test_quantity = quantity_tests[m]
        
        # not all combinations of seq and test quantities are valid
        if seq_quantity < test_quantity:
            out_folder = out_folder_preamble+'seqrate'+str(seq_iter)+'multiplier'+str(m)+'/results_density/'
            t_first_sequenced_temp = np.load(out_folder+'t_first_sequenced_all.npy')
            
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
                            e_cumulative_temp[a,b,c,:,d] = e_cumulative_temp_ts[a,b,c,:,int(the_time),d]
                            e_re_cumulative_temp[a,b,c,:,d] = e_re_cumulative_temp_ts[a,b,c,:,int(the_time),d]
        
            e_temp = e_cumulative_temp+e_re_cumulative_temp
        
            e_cumulative_all[m,seq_iter,:,:,:,:,:] = e_temp
            t_first_sequenced_all[m,seq_iter,:,:,:,:] = t_first_sequenced_temp

# save
np.save(out_folder_preamble+'combo_e_cumulative_all.npy', e_cumulative_all)
np.save(out_folder_preamble+'combo_t_first_sequenced_all.npy', t_first_sequenced_all)



