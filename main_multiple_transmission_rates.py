

## compare multiple transmissibility parameters

import numpy as np
import pandas as pd
from multiprocessing import Process

from simulations_functions import set_starting_compartments, loop_over_scenarios

import variants_parameters_more_transmissible as variant_pars



if __name__=='__main__':
    

    # set directories
    data_dir = './data'
    
    
    # --------------------------------------
    # ---------- Read data -----------------
    # --------------------------------------
    
    daily_tests = 7183

    # read data: NYC-reported test rates
    testrates = pd.read_csv(data_dir+'/nyc_testrates.csv')
    
    
    
    
    seqrates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]

    multipliers = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1., 1.2, 1.5, 2, 3, 4]
        
    b_values = [0.21, 0.25, 0.3, 0.35, 0.4, 0.5]

    # ------------------------------------------
    # ---------- Loop over beta
    # ------------------------------------------
    
    for b_iter in np.arange(len(b_values)):
        
        b2 = b_values[b_iter]
        b = [0.2, b2]
        
        for seq_iter in np.arange(len(seqrates)):
            
            seq_rate = seqrates[seq_iter]

	    	# set the same probability in all places
            p_gs = np.array([seq_rate]*len(testrates))

			# ------------------------------------------
			# ---------- Loop over test multipliers
			# ------------------------------------------

            for m in np.arange(len(multipliers)):
                
                multiplier = multipliers[m]
                # save in a separate folder for each seq rate
                out_foler_preamble = 'transmission_rate_multiples/bvalue_'+str(b_iter)+'_seqrate_'+str(seq_iter)+'_multiplier_'+str(m)+'/'
                # set the folder names
                out_folder1 = out_foler_preamble+'results_statusquo/'
                out_folder2 = out_foler_preamble+'results_density/'
                out_folder3 = out_foler_preamble+'results_random/'

                ### test rates
                # set base test rates to be the NYC average strategy
                test_rates = testrates.test_rate_percapita.values.copy()
                # adjust with multiplier
                test_rates = np.array([x*multiplier for x in test_rates])

                # density-based allocation (homogeneous) - borough
                test_rates_density_avg = np.array([int(daily_tests*multiplier)/np.sum(variant_pars.N)]*5)

                # random allocation (dictionary to randomly select new values for each iteration)
                random_testrates_params = {'pop_size_vector' : variant_pars.N, 'daily_tests' : int(daily_tests*multiplier)}


				# ----- simulation parameters
				# iterate over introduction location and introduction timing
				# note: currently fixed intro time points
				# later want to define the time point to match the peak/trough time

                T = 400
                n_iter = 100
                n_variants = 2
                n_locations = 5

    			# ----- setup for each run 

    			# initial compartments
                S, Sq, E, Iu, Itq, Itnq, Igq, Ignq, Ru, Rt, Rg, S_self, S_x, S_both = set_starting_compartments(n_variants, n_locations, variant_pars.N)
                
                compartments_inits = (S, Sq, E, Iu, Itq, Itnq, Igq, Ignq, Ru, Rt, Rg, S_self, S_x, S_both)

                intro_times = [0, 50, 80, 100, 120, 150]

    			# params with each of the test rates
                params1 = (variant_pars.fb_kmatrix, b, variant_pars.D, variant_pars.L, variant_pars.Q, test_rates, test_rates, p_gs, variant_pars.p_TP, variant_pars.p_FP, variant_pars.p_q, variant_pars.theta, variant_pars.w_self, variant_pars.w_x, variant_pars.a_self, variant_pars.a_cross, n_variants, n_locations)
                params2 = (variant_pars.fb_kmatrix, b, variant_pars.D, variant_pars.L, variant_pars.Q, test_rates_density_avg, test_rates_density_avg, p_gs, variant_pars.p_TP, variant_pars.p_FP, variant_pars.p_q, variant_pars.theta, variant_pars.w_self, variant_pars.w_x, variant_pars.a_self, variant_pars.a_cross, n_variants, n_locations)
                
                
                
                looping_params1 = {'T':T,
    				                'n_iter':n_iter,
    				                #'mini_batch_size':mini_batch_size,
    				                'compartments_inits':compartments_inits,
    				                'params':params1,
    				                 }
                
                looping_params2 = {'T':T,
    				                'n_iter':n_iter,
    				                #'mini_batch_size':mini_batch_size,
    				                'compartments_inits':compartments_inits,
    				                'params':params2,
    				                }



				# ------------------------------------------
				# ---------- Run the simulations -----------
				# ------------------------------------------

				### first the status quo strategy
            
                p = Process(target=loop_over_scenarios, args=(variant_pars.boroughs_order, intro_times,
                                                                out_folder1, looping_params1, 
    				                                            None, False))
                
                p.start()
                p.join()



    			### then the density based strategy
                
                p2 = Process(target=loop_over_scenarios, args=(variant_pars.boroughs_order, intro_times, 
    				                                            out_folder2, looping_params2, 
    				                                            None, False))
                p2.start()
                p2.join()
    				        


    			### then the random strategy
                
                p3 = Process(target=loop_over_scenarios, args=(variant_pars.boroughs_order, intro_times, 
    				                                            out_folder3, looping_params1, 
    				                                            random_testrates_params, False))
                p3.start()
                p3.join()





