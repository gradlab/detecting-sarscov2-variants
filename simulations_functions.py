import numpy as np
import copy
import pickle
import os
from numba import njit
from numba.typed import List




def make_test_rate_grid(n_samples, n_locations, pop_size_vector, daily_tests=50285):
    """generate an array of random test rates within constraints of max tests
    
    """
    sample_probs = np.zeros(shape=(n_samples, n_locations))
    # some random samples
    for i in np.arange(n_samples):
        # random sample of test rates
        rs = np.random.uniform(low=0., high=0.05, size=n_locations)
        # compute tests in raw terms
        rs_tests = rs*pop_size_vector
        # rescale so that the tests add up to exactly 50285
        rescaler = np.sum(rs_tests)/daily_tests
        new_rs_tests = rs_tests/rescaler
        # get the corresponding per capita test rates
        new_rs = new_rs_tests / pop_size_vector
        sample_probs[i,:] = new_rs

    return sample_probs


# ---------------------------------------------------------------------------------------



def make_k_matrix_from_M(M_matrix, avg_contacts):
    """generate a cotnact matrix from the raw mobility matrix and the average number of contacts per person
    
    Arguments:
        M_matrix     : number individuals moving from place in row to place in column
        avg_contacts : the nubmer of contacts per person
    Returns:
        k_matrix     : number of contacts between pairs of places (symmetric) 
    
    """
    assert M_matrix.shape[0]==M_matrix.shape[1]
    
    # number of locations
    nz = M_matrix.shape[0]
    
    # prep output
    k_matrix = np.zeros((nz, nz))
    
    # for each pair of zip codes / locations
    for z1 in np.arange(nz):
        for z2 in np.arange(nz):
            v=0
            # for each potential meeting place
            for place in np.arange(nz):
                move_from_z1_to_place = M_matrix[z1,place]
                move_from_z2_to_place = M_matrix[z2,place]
                # remove self-contact, i.e. subtract one from one of the diagonal entries
                if z1==z2:
                    move_from_z2_to_place -= 1
                all_moves_to_place = np.sum(M_matrix[:,place])

                v_temp = move_from_z1_to_place * avg_contacts * move_from_z2_to_place/all_moves_to_place 
                # (# individuals from z1 moving to place) * * (avg # contacts per person) 
                # * probabability that each person is an individual from z2 
                
                # add to the result
                v += v_temp

            k_matrix[z2,z1] = v
    return k_matrix



# ---------------------------------------------------------------------------------------



def set_starting_compartments(n_variants, n_locations, N):
    """set up starting conditions: all individuals in S compartment, other compartments are zero
    Arguments:
        n_variants  : number of variants in the model
        n_locations : number of patches / locations
        N           : population size
    
    Returns:
        initial compartments
    """
    
    Sq = np.zeros(shape=n_locations)
    E = np.zeros(shape=(n_variants,n_locations))
    Iu = np.zeros(shape=(n_variants,n_locations))
    Itq = np.zeros(shape=(n_variants,n_locations))
    Itnq = np.zeros(shape=(n_variants,n_locations))
    Igq = np.zeros(shape=(n_variants,n_locations))
    Ignq = np.zeros(shape=(n_variants,n_locations))
    Ru = np.zeros(shape=(n_variants,n_locations))
    Rt = np.zeros(shape=(n_variants,n_locations))
    Rg = np.zeros(shape=(n_variants,n_locations))
    S_self = np.zeros(shape=(n_variants,n_locations))
    S_x = np.zeros(shape=(n_variants,n_locations))
    S_both = np.zeros(shape=(n_variants,n_locations))
    
    S=np.array(N)-(np.sum((Iu+Itq+Itnq+Igq+Ignq+Ru+Rt+Rg+S_self+S_x+S_both), axis=0)+Sq)

    
    return S, Sq, E, Iu, Itq, Itnq, Igq, Ignq, Ru, Rt, Rg, S_self, S_x, S_both


# ----------------------------------------------------------
       



def simulate_later_introduction(T, n_iter, t_intro, intro_value, intro_loc, compartments_inits, params, random_testrates_params=None):
    
    """generate multiple simulations of the stochastic epidemic model
        given by the SEIR equations in fct
        where each variant may be introduced at different time points
    
    Arguments:
        T                   : time series length
        n_iter              : number of simulation iterations
        t_intro             : dictionary indicating when each (new) variant is introduced
        intro_value         : number of cases that are introduced at t_intro
        intro_loc           : location(s) where the variant is introduced
        compartments_inits  : list of initial values of compartment sizes
        params              : list of parameter values to be passed to fct
        random_testrates_params : parameters for random testing strategy

    Returns:
        out_dict            : dictionary with arrays for each compartment; 
                              array shape (n_variants,n_locations,T,n_iter)
    """
        
    n_variants = compartments_inits[2].shape[0]
    n_locations = compartments_inits[2].shape[1]

    # prep output arrays
    # time to first detection
    t_first_sequenced = np.zeros(shape=(n_variants, n_iter))
    # place of first detection
    patch_first_sequenced = np.zeros(shape=(n_variants, n_locations, n_iter))
    # cumulative compartments at all time points
    e_cumulative = np.zeros(shape=(n_variants, n_locations, T, n_iter))
    e_re_cumulative = np.zeros(shape=(n_variants, n_locations, T, n_iter))
    iu_cumulative = np.zeros(shape=(n_variants, n_locations, T, n_iter))
    it_cumulative = np.zeros(shape=(n_variants, n_locations, T, n_iter))
    ig_cumulative = np.zeros(shape=(n_variants, n_locations, T, n_iter))
    # time series of current I's
    Iu_series = np.zeros(shape=( n_variants, n_locations, T, n_iter))
    It_series = np.zeros(shape=( n_variants, n_locations, T, n_iter))
    Ig_series = np.zeros(shape=( n_variants, n_locations, T, n_iter))
    
    
    for iteration in np.arange(n_iter):
        
        # check for random option
        if random_testrates_params!=None:
            # random sample for test rates (within constraints)
            test_rates_random = make_test_rate_grid(n_samples=1, 
                                                    n_locations=n_locations, 
                                                    pop_size_vector=random_testrates_params['pop_size_vector'], 
                                                    daily_tests=random_testrates_params['daily_tests']).flatten().tolist()
            # update the parameters for the simulation
            fb_kmatrix, b, D, L, Q, p_ta, test_rates, p_gs, p_TP, p_FP, p_q, theta, w_self, w_x, a_self, a_cross, n_variants, n_locations = params
            params = (fb_kmatrix, b, D, L, Q, p_ta, test_rates_random, p_gs, p_TP, p_FP, p_q, theta, w_self, w_x, a_self, a_cross, n_variants, n_locations)


        # at the start of each iteration, set compartments to init values
        compartments_new = copy.deepcopy(compartments_inits)
        
        
        # reset the temp variables for computing outcome measures
        t_found = np.zeros(shape=n_variants)
        
        # loop over time steps
        for t in np.arange(T):

            # save the time series for the I compartments only
            Iu_series[:,:,t,iteration] = compartments_new[3].tolist()
            It_series[:,:,t,iteration] = compartments_new[4].tolist()
            Ig_series[:,:,t,iteration] = compartments_new[5].tolist()
            
            
            
            # check if a new variant is introduced
            for v in np.arange(n_variants):
                # if yes, update the Iu value accordingly
                if v in t_intro.keys():
                    if t_intro[v]==t:
                        zs = intro_loc[v]
                        for z in zs:
                            compartments_new[3][v,z]=intro_value[v]

            # compute the next step
            compartments_new, cumulative_next = seir_multiple_variants_NUMBA(compartments_new, params)
            
            # 2) update the output measures
            # 2.1) cumulative E, I, R
            dn_s_e, dn_sx_eother_destination, dn_sboth_eother_destination, dn_sself_eself, dn_sboth_eself, dn_e_iu, dn_e_itq, dn_e_itnq, dn_e_igq, dn_e_ignq, dn_iu_ru, dn_itq_rt, dn_itnq_rt, dn_igq_rg, dn_ignq_rg = cumulative_next
            
            
            
            # while t<t_first_sequenced
            # add dn's to the cumulative functions
            
            # update the cumulative values (only until the variant is sequenced)
            if t==0:
                e_cumulative[v,:,0,iteration] += dn_s_e[v,:]
                e_re_cumulative[v,:,0,iteration] += (dn_sx_eother_destination[v,:] + dn_sboth_eother_destination[v,:] + dn_sself_eself[v,:] + dn_sboth_eself[v,:])
                iu_cumulative[v,:,0,iteration] += (dn_e_iu[v,:])
                it_cumulative[v,:,0,iteration] += (dn_e_itq[v,:] + dn_e_itnq[v,:])
                ig_cumulative[v,:,0,iteration] += (dn_e_igq[v,:] + dn_e_ignq[v,:])
                #r_cumulative[v,:,0,iteration] += (dn_iu_ru[v,:] + dn_itq_rt[v,:] + dn_itnq_rt[v,:] + dn_igq_rg[v,:] + dn_ignq_rg[v,:])
            if t>0:
                e_cumulative[v,:,t,iteration] = e_cumulative[v,:,t-1,iteration] + dn_s_e[v,:]
                e_re_cumulative[v,:,t,iteration] = e_re_cumulative[v,:,t-1,iteration] + (dn_sx_eother_destination[v,:] + dn_sboth_eother_destination[v,:] + dn_sself_eself[v,:] + dn_sboth_eself[v,:])
                iu_cumulative[v,:,t,iteration] = iu_cumulative[v,:,t-1,iteration] + (dn_e_iu[v,:])
                it_cumulative[v,:,t,iteration] = it_cumulative[v,:,t-1,iteration] + (dn_e_itq[v,:] + dn_e_itnq[v,:])
                ig_cumulative[v,:,t,iteration] = ig_cumulative[v,:,t-1,iteration] + (dn_e_igq[v,:] + dn_e_ignq[v,:])
                #r_cumulative[v,:,t,iteration] = r_cumulative[v,:,t-1,iteration] + (dn_iu_ru[v,:] + dn_itq_rt[v,:] + dn_itnq_rt[v,:] + dn_igq_rg[v,:] + dn_ignq_rg[v,:])
                 
            
            # 2.2) t first sequenced
            # &
            # 2.3) location first sequenced
            
            for v in np.arange(n_variants):
                # if variant is not yet sequenced, compute the cumulative
                if t_found[v]==0: 
                        
                    first_patch_fool=[]
                    for z in np.arange(n_locations):
                        if ((dn_e_igq + dn_e_ignq)[v,z]>=1):
                            # set t_first_sequenced
                            t_first_sequenced[v,iteration] = t
                            # add corresponding patch(es) to a list
                            first_patch_fool.append(z)
                            # update t_found, so that next time steps are not updated
                            t_found[v] = 1
                    # if some patches were sequenced:
                    if len(first_patch_fool)>0:
                        # update the dictionary linking the variant to the patch(es) first sequenced
                        patch_first_sequenced[v,first_patch_fool,iteration] = 1
            
                # edge case: variant never sequenced --> zeros
                
                    
                    
    return t_first_sequenced, patch_first_sequenced, e_cumulative, e_re_cumulative, iu_cumulative, it_cumulative, ig_cumulative, Iu_series, It_series, Ig_series
    





# ----------------------------------------------------------





def loop_over_scenarios(boroughs_order, intro_times, out_folder, looping_params, random_testrates_params, save_intermediate=False):
    
    """loop over intro times and start boroughs
    
    Arguments:
        boroughs_order  : list of locations
        intro_times     : list of time points to loop over for introduction of second variant
        out_folder      : directory to save results
        looping_params  : dictionary with required input params, specifically T, n_iter, compartments_inits, params
        random_testrates_params : parameters for random test allocation strategy
        save_intermediate : boolean, whether to save output intermittently
    
    Returns:
        None
    """
    

    T = looping_params['T']
    n_iter = looping_params['n_iter']
    compartments_inits = looping_params['compartments_inits']
    params = looping_params['params']
    
    n_variants = compartments_inits[2].shape[0]
    n_locations = compartments_inits[2].shape[1]


	# outcome measures
    t_first_sequenced_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_iter))
    patch_first_sequenced_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, n_iter))
    # cumulative (over time, not just at detection time)
    e_cumulative_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    e_re_cumulative_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    iu_cumulative_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    it_cumulative_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    ig_cumulative_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    # series
    Iu_series_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    It_series_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    Ig_series_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    
    
    # check if output folder exists, if not, make it:
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # save the parameter values for reference
    a_file = out_folder+'parameters.pkl'
    with open(a_file, 'wb') as handle:
        pickle.dump(params, handle)
    # and the order of the first and second dimensions (intro location and time)
    a_file = out_folder+'intro_locations.pkl'
    with open(a_file, 'wb') as handle:
        pickle.dump(boroughs_order, handle)

    a_file = out_folder+'intro_times.pkl'
    with open(a_file, 'wb') as handle:
        pickle.dump(intro_times, handle)
    

    for z_index, index_location in enumerate(boroughs_order):

        for t_index, t in enumerate(intro_times):

            # set the introduction parameters
            t_intro = {0:0, 1:t} # introduce variant 1 at time t
            intro_val = {0:1, 1:1} # introduce a single case (per location)
            intro_loc = {0:[0,1,2,3,4], 1:[z_index]} # introduce one case in each borough

            # -----------------
            # run the simulations
            t_first_sequenced, patch_first_sequenced, e_cumulative, e_re_cumulative, iu_cumulative, it_cumulative, ig_cumulative, Iu_series, It_series, Ig_series = simulate_later_introduction(T=T, n_iter=n_iter, t_intro=t_intro, 
            																																						intro_value=intro_val, intro_loc=intro_loc, 
                                                                      																								compartments_inits=compartments_inits, 
                                                                      																								params=params, random_testrates_params=random_testrates_params)
			
    
            
			# -----------------
            # outcome measures
            t_first_sequenced_all[z_index, t_index, :, :] = t_first_sequenced
            patch_first_sequenced_all[z_index, t_index, :, :, :] = patch_first_sequenced
            # cumulatives
            e_cumulative_all[z_index, t_index, :, :, :, :] = e_cumulative
            e_re_cumulative_all[z_index, t_index, :, :, :, :] = e_re_cumulative
            iu_cumulative_all[z_index, t_index, :, :, :, :] = iu_cumulative
            it_cumulative_all[z_index, t_index, :, :, :, :] = it_cumulative
            ig_cumulative_all[z_index, t_index, :, :, :, :] = ig_cumulative
		    # time series
            Iu_series_all[z_index, t_index, :, :, :, :] = Iu_series
            It_series_all[z_index, t_index, :, :, :, :] = It_series
            Ig_series_all[z_index, t_index, :, :, :, :] = Ig_series
            
            
            # possibly save intermediate values
            if save_intermediate==True:

                # output measures
                np.save(out_folder+'t_first_sequenced_all.npy', t_first_sequenced_all)
                np.save(out_folder+'patch_first_sequenced_all.npy', patch_first_sequenced_all)
                np.save(out_folder+'e_cumulative_all_ts.npy', e_cumulative_all)
                np.save(out_folder+'e_re_cumulative_all_ts.npy', e_re_cumulative_all)
                np.save(out_folder+'iu_cumulative_all_ts.npy', iu_cumulative_all)
                np.save(out_folder+'it_cumulative_all_ts.npy', it_cumulative_all)
                np.save(out_folder+'ig_cumulative_all_ts.npy', ig_cumulative_all)
                #np.save(out_folder+'r_cumulative_all.npy', r_cumulative_all)



    # -----------------
    # write data to file

    # output measures
    np.save(out_folder+'t_first_sequenced_all.npy', t_first_sequenced_all)
    np.save(out_folder+'patch_first_sequenced_all.npy', patch_first_sequenced_all)
    np.save(out_folder+'e_cumulative_all_ts.npy', e_cumulative_all)
    np.save(out_folder+'e_re_cumulative_all_ts.npy', e_re_cumulative_all)
    np.save(out_folder+'iu_cumulative_all_ts.npy', iu_cumulative_all)
    np.save(out_folder+'it_cumulative_all_ts.npy', it_cumulative_all)
    np.save(out_folder+'ig_cumulative_all_ts.npy', ig_cumulative_all)
    #np.save(out_folder+'r_cumulative_all.npy', r_cumulative_all)

        
# ------------------------------------------------------    

# fixed resources



def loop_over_scenarios_fixedresources(boroughs_order, intro_times, out_folder, looping_params, seq_quantity,
                                                random_testrates_params,
                                                save_intermediate=False):
    
    """loop over intro times and start boroughs
    
    Arguments:
        boroughs_order  : list of locations
        intro_times     : list of time points to loop over for introduction of second variant
        out_folder      : directory to save results
        looping_params  : dictionary with required input params, specifically T, n_iter, compartments_inits, params
        seq_quantity    : number of sequences
        random_testrates_params : parameters to set up random testing
        save_intermediate : boolean, whether to save output intermittently
    
    Returns:
        None
    """
    

    T = looping_params['T']
    n_iter = looping_params['n_iter']
    compartments_inits = looping_params['compartments_inits']
    params = looping_params['params']
    
    n_variants = compartments_inits[2].shape[0]
    n_locations = compartments_inits[2].shape[1]


    # outcome measures
    t_first_sequenced_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_iter))
    patch_first_sequenced_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, n_iter))
    # cumulative (over time, not just at detection time)
    e_cumulative_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    e_re_cumulative_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    iu_cumulative_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    it_cumulative_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    ig_cumulative_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    # series
    Iu_series_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    It_series_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    Ig_series_all = np.zeros(shape=(len(boroughs_order), len(intro_times),n_variants, n_locations, T, n_iter))
    
    
    # check if output folder exists, if not, make it:
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # save the parameter values for reference
    a_file = out_folder+'parameters.pkl'
    with open(a_file, 'wb') as handle:
        pickle.dump(params, handle)
    # and the order of the first and second dimensions (intro location and time)
    a_file = out_folder+'intro_locations.pkl'
    with open(a_file, 'wb') as handle:
        pickle.dump(boroughs_order, handle)

    a_file = out_folder+'intro_times.pkl'
    with open(a_file, 'wb') as handle:
        pickle.dump(intro_times, handle)
    

    for z_index, index_location in enumerate(boroughs_order):

        for t_index, t in enumerate(intro_times):

            # set the introduction parameters
            t_intro = {0:0, 1:t} # introduce variant 1 at time t
            intro_val = {0:1, 1:1} # introduce a single case (per location)
            intro_loc = {0:[0,1,2,3,4], 1:[z_index]} # introduce one case in each borough

            # -----------------
            # run the simulations
            
            t_first_sequenced, patch_first_sequenced, e_cumulative, e_re_cumulative, iu_cumulative, it_cumulative, ig_cumulative, Iu_series, It_series, Ig_series = simulate_later_introduction_fixedresources(T=T, n_iter=n_iter, t_intro=t_intro, 
                                                                                                                                                                    intro_value=intro_val, intro_loc=intro_loc, 
                                                                                                                                                                    compartments_inits=compartments_inits, 
                                                                                                                                                                    params=params, 
                                                                                                                                                                    seq_quantity = seq_quantity,
                                                                                                                                                                    random_testrates_params=random_testrates_params)
            
    
            
            # -----------------
            # combine data before writing to file
            
            # outcome measures
            t_first_sequenced_all[z_index, t_index, :, :] = t_first_sequenced
            patch_first_sequenced_all[z_index, t_index, :, :, :] = patch_first_sequenced
            # cumulatives
            e_cumulative_all[z_index, t_index, :, :, :, :] = e_cumulative
            e_re_cumulative_all[z_index, t_index, :, :, :, :] = e_re_cumulative
            iu_cumulative_all[z_index, t_index, :, :, :, :] = iu_cumulative
            it_cumulative_all[z_index, t_index, :, :, :, :] = it_cumulative
            ig_cumulative_all[z_index, t_index, :, :, :, :] = ig_cumulative
            # time series
            Iu_series_all[z_index, t_index, :, :, :, :] = Iu_series
            It_series_all[z_index, t_index, :, :, :, :] = It_series
            Ig_series_all[z_index, t_index, :, :, :, :] = Ig_series
            
            
            # possibly save intermediate values
            if save_intermediate==True:
                
                # output measures
                np.save(out_folder+'t_first_sequenced_all.npy', t_first_sequenced_all)
                np.save(out_folder+'patch_first_sequenced_all.npy', patch_first_sequenced_all)
                np.save(out_folder+'e_cumulative_all_ts.npy', e_cumulative_all)
                np.save(out_folder+'e_re_cumulative_all_ts.npy', e_re_cumulative_all)
                np.save(out_folder+'iu_cumulative_all_ts.npy', iu_cumulative_all)
                np.save(out_folder+'it_cumulative_all_ts.npy', it_cumulative_all)
                np.save(out_folder+'ig_cumulative_all_ts.npy', ig_cumulative_all)
                #np.save(out_folder+'r_cumulative_all.npy', r_cumulative_all)



    # -----------------
    # write data to file

    # output measures
    np.save(out_folder+'t_first_sequenced_all.npy', t_first_sequenced_all)
    np.save(out_folder+'patch_first_sequenced_all.npy', patch_first_sequenced_all)
    np.save(out_folder+'e_cumulative_all_ts.npy', e_cumulative_all)
    np.save(out_folder+'e_re_cumulative_all_ts.npy', e_re_cumulative_all)
    np.save(out_folder+'iu_cumulative_all_ts.npy', iu_cumulative_all)
    np.save(out_folder+'it_cumulative_all_ts.npy', it_cumulative_all)
    np.save(out_folder+'ig_cumulative_all_ts.npy', ig_cumulative_all)
    #np.save(out_folder+'r_cumulative_all.npy', r_cumulative_all)



# --------
def simulate_later_introduction_fixedresources(T, n_iter, t_intro, intro_value, intro_loc, compartments_inits, params, seq_quantity,
                                                        random_testrates_params=None):
    """generate multiple simulations of the stochastic epidemic model
        given by the SEIR equations in fct
        where each variant may be introduced at different time points
    
    Arguments:
        T                   : time series length
        n_iter              : number of simulation iterations
        t_intro             : dictionary indicating when each (new) variant is introduced
        intro_value         : number of cases that are introduced at t_intro
        intro_loc           : location(s) where the variant is introduced
        compartments_inits  : list of initial values of compartment sizes
        params              : list of parameter values to be passed to fct
        seq_quantity        : number of sequences
        random_testrates_params : parameters for random test allocation strategy

    Returns:
        out_dict            : dictionary with arrays for each compartment; 
                              array shape (n_variants,n_locations,T,n_iter)
    """
    
    
    n_variants = compartments_inits[2].shape[0]
    n_locations = compartments_inits[2].shape[1]


    # time to first detection
    t_first_sequenced = np.zeros(shape=(n_variants, n_iter))
    # place of first detection
    patch_first_sequenced = np.zeros(shape=(n_variants, n_locations, n_iter))
    # cumulative compartments at all time points
    e_cumulative = np.zeros(shape=(n_variants, n_locations, T, n_iter))
    e_re_cumulative = np.zeros(shape=(n_variants, n_locations, T, n_iter))
    iu_cumulative = np.zeros(shape=(n_variants, n_locations, T, n_iter))
    it_cumulative = np.zeros(shape=(n_variants, n_locations, T, n_iter))
    ig_cumulative = np.zeros(shape=(n_variants, n_locations, T, n_iter))
    # time series of current I's
    Iu_series = np.zeros(shape=( n_variants, n_locations, T, n_iter))
    It_series = np.zeros(shape=( n_variants, n_locations, T, n_iter))
    Ig_series = np.zeros(shape=( n_variants, n_locations, T, n_iter))
    
    
    
    for iteration in np.arange(n_iter):
        
        # check for random option
        if random_testrates_params!=None:
            # random sample for test rates (within constraints)
            test_rates_random = make_test_rate_grid(n_samples=1, 
                                                    n_locations=n_locations, 
                                                    pop_size_vector=random_testrates_params['pop_size_vector'], 
                                                    daily_tests=random_testrates_params['daily_tests']).flatten().tolist()
            # update the parameters for the simulation
            fb_kmatrix, b, D, L, Q, p_ta, test_rates, p_gs, p_TP, p_FP, p_q, theta, w_self, w_x, a_self, a_cross, n_variants, n_locations = params
            params = (fb_kmatrix, b, D, L, Q, p_ta, test_rates_random, p_gs, p_TP, p_FP, p_q, theta, w_self, w_x, a_self, a_cross, n_variants, n_locations)


        # at the start of each iteration, set compartments to init values
        compartments_new = copy.deepcopy(compartments_inits)
        
        
        # reset the temp variables for computing outcome measures
        t_found = np.zeros(shape=n_variants)
        
        # loop over time steps
        for t in np.arange(T):
            
            # save the time series for the I compartments only
            Iu_series[:,:,t,iteration] = compartments_new[3].tolist()
            It_series[:,:,t,iteration] = compartments_new[4].tolist()
            Ig_series[:,:,t,iteration] = compartments_new[5].tolist()
            
            
            # check if a new variant is introduced
            for v in np.arange(n_variants):
                # if yes, update the Iu value accordingly
                if v in t_intro.keys():
                    if t_intro[v]==t:
                        zs = intro_loc[v]
                        for z in zs:
                            compartments_new[3][v,z]=intro_value[v]

            # compute the next step
            compartments_new, cumulative_next = seir_multiple_variants_NUMBA(compartments_new, params)
            
            # 2) update the output measures
            # 2.1) cumulative E, I, R
            dn_s_e, dn_sx_eother_destination, dn_sboth_eother_destination, dn_sself_eself, dn_sboth_eself, dn_e_iu, dn_e_itq, dn_e_itnq, dn_e_igq, dn_e_ignq, dn_iu_ru, dn_itq_rt, dn_itnq_rt, dn_igq_rg, dn_ignq_rg = cumulative_next
            
            
            
            # while t<t_first_sequenced
            # add dn's to the cumulative functions
            
            if t==0:
                e_cumulative[v,:,0,iteration] += dn_s_e[v,:]
                e_re_cumulative[v,:,0,iteration] += (dn_sx_eother_destination[v,:] + dn_sboth_eother_destination[v,:] + dn_sself_eself[v,:] + dn_sboth_eself[v,:])
                iu_cumulative[v,:,0,iteration] += (dn_e_iu[v,:])
                it_cumulative[v,:,0,iteration] += (dn_e_itq[v,:] + dn_e_itnq[v,:])
                ig_cumulative[v,:,0,iteration] += (dn_e_igq[v,:] + dn_e_ignq[v,:])
                #r_cumulative[v,:,0,iteration] += (dn_iu_ru[v,:] + dn_itq_rt[v,:] + dn_itnq_rt[v,:] + dn_igq_rg[v,:] + dn_ignq_rg[v,:])
            if t>0:
                e_cumulative[v,:,t,iteration] = e_cumulative[v,:,t-1,iteration] + dn_s_e[v,:]
                e_re_cumulative[v,:,t,iteration] = e_re_cumulative[v,:,t-1,iteration] + (dn_sx_eother_destination[v,:] + dn_sboth_eother_destination[v,:] + dn_sself_eself[v,:] + dn_sboth_eself[v,:])
                iu_cumulative[v,:,t,iteration] = iu_cumulative[v,:,t-1,iteration] + (dn_e_iu[v,:])
                it_cumulative[v,:,t,iteration] = it_cumulative[v,:,t-1,iteration] + (dn_e_itq[v,:] + dn_e_itnq[v,:])
                ig_cumulative[v,:,t,iteration] = ig_cumulative[v,:,t-1,iteration] + (dn_e_igq[v,:] + dn_e_ignq[v,:])
                #r_cumulative[v,:,t,iteration] = r_cumulative[v,:,t-1,iteration] + (dn_iu_ru[v,:] + dn_itq_rt[v,:] + dn_itnq_rt[v,:] + dn_igq_rg[v,:] + dn_ignq_rg[v,:])
                 
            
            # 2.2) t first sequenced
            # &
            # 2.3) location first sequenced
            
            for v in np.arange(n_variants):
                # if variant is not yet sequenced, compute the cumulative
                if t_found[v]==0:   
                        
                    first_patch_fool=[]
                    for z in np.arange(n_locations):
                        if ((dn_e_igq + dn_e_ignq)[v,z]>=1):
                            # set t_first_sequenced
                            t_first_sequenced[v,iteration] = t
                            # add corresponding patch(es) to a list
                            first_patch_fool.append(z)
                            # update t_found, so that next time steps are not updated
                            t_found[v] = 1
                    # if some patches were sequenced:
                    if len(first_patch_fool)>0:
                        # update the dictionary linking the variant to the patch(es) first sequenced
                        patch_first_sequenced[v,first_patch_fool,iteration] = 1
            
                # edge case: variant never sequenced --> zeros
                
            ## after each time step, update the sequencing probability
            seq_quantity_loc = [seq_quantity/n_locations] * n_locations
            num_Es_loc = np.sum(compartments_new[2], axis=0) # the E compartment, shape: (n_variants,n_locations) -> sum over variants
            p_gs_new = np.array(seq_quantity_loc) / np.array(num_Es_loc)
            # cap at 1.0
            p_gs_new[p_gs_new > 1.0] = 1.0
            
            print(p_gs_new)

            # update params:
            fb_kmatrix, b, D, L, Q, p_ta, p_ts, p_gs, p_TP, p_FP, p_q, theta, w_self, w_x, a_self, a_cross, n_variants, n_locations = params
            params = (fb_kmatrix, b, D, L, Q, p_ta, p_ts, p_gs_new, p_TP, p_FP, p_q, theta, w_self, w_x, a_self, a_cross, n_variants, n_locations)

                    
    return t_first_sequenced, patch_first_sequenced, e_cumulative, e_re_cumulative, iu_cumulative, it_cumulative, ig_cumulative, Iu_series, It_series, Ig_series
    








