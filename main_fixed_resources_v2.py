

# fixed resources

import numpy as np
import pandas as pd
from multiprocessing import Process


from simulations_functions import set_starting_compartments, loop_over_scenarios_fixedresources

import variants_parameters_more_transmissible as variant_pars



if __name__=='__main__':
    

    # set directories
    data_dir = './data'
    
    
    # --------------------------------------
    # ---------- Read data -----------------
    # --------------------------------------
    
    daily_tests = 7183

    quantity_tests = [500, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10500, 12500, 14500, 16500]
    quantity_sequences = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4500, 5500, 6500, 7500, 9000, 10500, 12000, 13500, 15000]


    # for statusquo, get the proportion of tests allocated to each borough
    testrates = pd.read_csv(data_dir+'/nyc_testrates.csv') 
    
    
        
    
    # --------------------------------------------------------------
    # ---------- Loop over sequencing and test quantities
    # --------------------------------------------------------------
    

    b = [0.2, 0.3]
    
    # loop: seq quantities
    for seq_iter in np.arange(len(quantity_sequences)):
        
        seq_quantity = quantity_sequences[seq_iter]

        # loop: test quantities
        for m in np.arange(len(quantity_tests)):

            test_quantity = quantity_tests[m]

            # check that seq quantity is less than test quantity
            # otherwise, skip the combo of parameters
            if seq_quantity < test_quantity:

                # set p_gs and p_ts: allocate the tests and sequences according to pop. density
                # test rate fixed across boroughs
                test_rates_density_avg = np.array([test_quantity/np.sum(variant_pars.N)]*5)
                # sequencing rate is the share of sequences that is tested in each place
                # p_gs = np.array([seq_quantity/test_quantity]*5)
                # sequencing rate starts off as zero
                p_gs = np.array([0.0]*5)


                # save in a separate folder for each seq rate
                out_foler_preamble = 'fixed_resources/seqrate_'+str(seq_iter)+'_multiplier_'+str(m)+'/'
                # set the folder names
                out_folder2 = out_foler_preamble+'results_density/'


        		
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
                
                intro_times = [0, 50, 80, 100, 120, 150]
                
                compartments_inits = (S, Sq, E, Iu, Itq, Itnq, Igq, Ignq, Ru, Rt, Rg, S_self, S_x, S_both)

            	# params for simulations (only the density-based approach)
                params2 = (variant_pars.fb_kmatrix, b, variant_pars.D, variant_pars.L, variant_pars.Q, test_rates_density_avg, test_rates_density_avg, p_gs, variant_pars.p_TP, variant_pars.p_FP, variant_pars.p_q, variant_pars.theta, variant_pars.w_self, variant_pars.w_x, variant_pars.a_self, variant_pars.a_cross, n_variants, n_locations)
                
                
                
                looping_params2 = {'T':T,
        			                  'n_iter':n_iter,
        			                  #'mini_batch_size':mini_batch_size,
        			                  'compartments_inits':compartments_inits,
        			                  'params':params2,
        			                 }



    			# ------------------------------------------
    			# ---------- Run the simulations -----------
    			# ------------------------------------------


        		### the density based strategy
                
                p2 = Process(target=loop_over_scenarios_fixedresources, args=(variant_pars.boroughs_order, intro_times, 
        			                                                             out_folder2, looping_params2, seq_quantity, 
        			                                                             None, False))
                p2.start()
                p2.join()
        			        

               


