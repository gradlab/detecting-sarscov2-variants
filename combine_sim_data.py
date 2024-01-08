


############################################################## 
# prep data for viz & analysis: combine sim results 
############################################################## 


import numpy as np
import os




# set location of individual files
out_folder_preamble = 'lower-seq-rates/transmission_rate_multiples_updated-daily/'


# set the dimensions of the individual data files, based on len of parameter values

strategies = ['statusquo', 'density', 'random']
seqrates = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
multipliers = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1., 1.2, 1.5, 2, 3, 4]
bvalues = [0.21, 0.25, 0.3, 0.35, 0.4, 0.5]
intro_locations = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten']
intro_times = [0, 50, 80, 100, 120, 150]

n_strategies= len(strategies)
n_multipliers = len(multipliers)
n_seqrates = len(seqrates)
n_bvalues = len(bvalues)
n_variants = 2
n_iter = 100


# set the outputs
e_cumulative_all = np.zeros(shape=( n_bvalues, n_strategies, n_multipliers, 
                                   n_seqrates, len(intro_locations), len(intro_times),
                                   n_variants, len(intro_locations), n_iter) )
t_first_sequenced_all = np.zeros(shape=( n_bvalues, n_strategies,n_multipliers, 
                                   n_seqrates, len(intro_locations), len(intro_times),
                                   n_variants, n_iter) )


for ff, stratname in enumerate(strategies):
    for bb in np.arange(n_bvalues):
        for mm in np.arange(n_multipliers): 
            for ss in np.arange(n_seqrates):
                
                out_folder = out_folder_preamble+'bvalue_'+str(bb)+'seqrate'+str(ss)+'multiplier'+str(mm)+'/results_'+stratname+'/'
    			
                # time to detection
                t_first_sequenced_temp = np.load(out_folder+'t_first_sequenced_all.npy')
              
                # detection locations
                patch_first_sequenced_temp = np.load(out_folder+'patch_first_sequenced_all.npy')


                # cumulative exposures, re-exposures
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
                
            
                e_cumulative_all[bb,ff,mm,ss,:,:,:,:,:] = e_temp
                t_first_sequenced_all[bb,ff,mm,ss,:,:,:,:] = t_first_sequenced_temp
                

# save
np.save(out_folder_preamble+'combo_e_cumulative_all.npy', e_cumulative_all)
np.save(out_folder_preamble+'combo_t_first_sequenced_all.npy', t_first_sequenced_all)








