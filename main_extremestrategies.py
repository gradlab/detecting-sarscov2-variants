
# extreme allocations of tests

import numpy as np
import pandas as pd
from multiprocessing import Process


from simulations_functions import set_starting_compartments, loop_over_scenarios

import variants_parameters_more_transmissible as variant_pars



if __name__=='__main__':
    
    # ----------------------------------------------------------------------------------
    # load required data
    # ----------------------------------------------------------------------------------
    
    data_dir = './data'
    
    # nyc population
    borough_pop = pd.read_csv(data_dir+'/borough_population.csv')
    
    n_variants = 2
    n_locations = 5
    
    
    # ----------------------------------------------------------------------------------
    # set up the grid of test rates
    # ----------------------------------------------------------------------------------
    
    
    # ------ variable parameters
    daily_tests = 7183
    n_locations = 5
    pop_size_vector = np.array(borough_pop.pop_2020_estimate)
    # ------
    
    
    for e, extreme_prop in enumerate([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):    
        
        # extreme proportion allocated to a single borough
        extremes_tests = np.identity(n=5) * daily_tests * extreme_prop

        # remaining tests allocated evenly (by pop density)
        remaining_tests = (1-extreme_prop) * daily_tests
        remaining_alloc = np.zeros(shape=extremes_tests.shape)
        for i in np.arange(extremes_tests.shape[0]):
            N_temp = np.array(variant_pars.N).copy()
            # replace the given loc with zero, so that no more tests are allocated there
            N_temp[i] = 0
            pop_proportion = N_temp / np.sum(N_temp)

            remaining_alloc[i,:] = remaining_tests * pop_proportion

        # rates are computed as total tests (extremes+remaining) divided by pop size
        extremes_rates = (extremes_tests + remaining_alloc) / np.array(variant_pars.N)
        
        
        for i in np.arange(extremes_rates.shape[0]):

            test_rates = np.array(extremes_rates[i,:])
            ### sequencing rate 10%
            p_gs = np.array([0.1]*5)
        
        
            T = 400
            n_iter = 100
            n_variants = 2
            n_locations = 5
            
            # ----- setup for each run 
            
            # initial compartments
            S, Sq, E, Iu, Itq, Itnq, Igq, Ignq, Ru, Rt, Rg, S_self, S_x, S_both = set_starting_compartments(n_variants, n_locations, np.array(variant_pars.N) )
            # probability of testing without symptoms assumed to be equal to the average per capita test rate
            p_ta = test_rates.copy()

            intro_times = [0, 50, 80, 100, 120, 150]
            
            compartments_inits = (S, Sq, E, Iu, Itq, Itnq, Igq, Ignq, Ru, Rt, Rg, S_self, S_x, S_both)

            # params with each of the test rates
            params1 = (variant_pars.fb_kmatrix, variant_pars.b, variant_pars.D, variant_pars.L, variant_pars.Q, p_ta, test_rates, variant_pars.p_gs, variant_pars.p_TP, variant_pars.p_FP, variant_pars.p_q, variant_pars.theta, variant_pars.w_self, variant_pars.w_x, variant_pars.a_self, variant_pars.a_cross, n_variants, n_locations)
            

            
            out_folder1 = 'extreme_strategies/results_extreme-prop-'+str(e)+'_extreme-loc-'+str(i)+'/'
            
            
            looping_params1 = {'T':T,
                              'n_iter':n_iter,
                              #'mini_batch_size':mini_batch_size,
                              'compartments_inits':compartments_inits,
                              'params':params1,
                             }
            
            

            
            
            # ------------------------------------------
            # ---------- Run the simulations -----------
            # ------------------------------------------

            ### status quo strategy
            
            p = Process(target=loop_over_scenarios, args=(variant_pars.boroughs_order, intro_times, 
                                                            out_folder1, looping_params1, 
                                                            None, False))            
            p.start()
            p.join()
            
            
            