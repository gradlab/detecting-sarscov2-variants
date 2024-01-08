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
       

@njit(debug=True)
def seir_multiple_variants_NUMBA(compartments_inits, params ):
    """generate the next step of the stochastic SEIR model    

    Arguments:
        compartments_inits  : list of initial values of compartment sizes
        params              : list of parameter values to be passed to fct

    Returns:
        next_compartments   : list of next compartment values, in same order as inits
        next_cumulative     : list of deltas in flows to compute cumulative I, E, and R
    """
    
    # expand inputs
    S, Sq, E, Iu, Itq, Itnq, Igq, Ignq, Ru, Rt, Rg, S_self, S_x, S_both = compartments_inits
    kmatrix, b, D, L, Q, p_ta, p_ts, p_gs, p_TP, p_FP, p_q, theta, w_self, w_x, a_self, a_cross, n_variants, n_locations = params
    
    # calculate pop size in each patch
    N = np.sum(E+Iu+Itq+Itnq+Igq+Ignq+Ru+Rt+Rg+S_self+S_x+S_both, axis=0)+S+Sq

    # rate of leaving quarantine/ ending contact reduction without symptoms
    epsilon = 1/Q
    

    # for later use, combine the different I compartments for each patch and variant
    # accounting for the reduced infectiousness of the individuals in quarantine / with reduced contacts
    # shape: (n_variants, n_patches)
    I_combo = (Iu+Itnq+Ignq) + (Itq+Igq) * theta
    
    
    ### ------- Flows out of S -------
    # exposures:
    lambda_param = np.zeros(shape=(n_variants, n_locations))
    for v in np.arange(n_variants):    
        for z in np.arange(n_locations):
            temp = 0
            # sum over all the (other) locations
            for j in np.arange(n_locations):
                temp += kmatrix[j,z]*I_combo[v,j]/N[j]

            lambda_param[v,z] = b[v]*(temp/N[z])

    # add Sq flow:
    p_leaving_S = np.zeros(shape=(lambda_param.shape[0]+1, lambda_param.shape[1]))
    p_leaving_S[:-1,:] = lambda_param
    p_leaving_S[-1,:] = p_ta*p_FP
    
    dn_s_e = np.zeros(shape=(n_variants, n_locations))
    # ... and the Sq compartment
    dn_s_sq = np.zeros(shape=n_locations)
    dn_sq_s = np.zeros(shape=n_locations)
    for z in np.arange(n_locations):
        # check
        assert np.sum(p_leaving_S[:,z])<=1
        
        # store the number of people leaving S for the E's, moving to Sq, and staying in a given place
        fool = np.random.multinomial(n=int(S[z]), 
                                     pvals=list(p_leaving_S[:,z]) + [1-np.sum(p_leaving_S[:,z])])
        dn_s_e[:,z] = fool[:-2]
        dn_s_sq[z] = fool[-2]

        ## flow from S_q back into S
        dn_sq_s[z] = np.random.binomial(n=int(Sq[z]), p=epsilon)
    
    
    ### ------- Flows out of E -------
    leaving_E = np.zeros(shape=(n_variants, n_locations))
    dn_e_iu = np.zeros(shape=(n_variants, n_locations))
    dn_e_itq = np.zeros(shape=(n_variants, n_locations))
    dn_e_itnq = np.zeros(shape=(n_variants, n_locations))
    dn_e_igq = np.zeros(shape=(n_variants, n_locations))
    dn_e_ignq = np.zeros(shape=(n_variants, n_locations))

    for v in np.arange(n_variants):
        for z in np.arange(n_locations):
            leaving_E[v,z] = np.random.binomial(n=int(E[v,z]), p=1/L[v])

            # divide the leaving E among the I compartments 
            # separate sampling steps: 
            # 1) tested / not tested
            # 2) positive (TP) or negative (FN) test result
            # 3) if tested: sequenced / not sequenced
            # 4) if each tested or tested & sequenced, quarantined or not

            # 1) number tested
            e_tested = np.random.binomial(n=int(leaving_E[v,z]), p=p_ts[z])
            # 2) TP vs FN
            e_TP = np.random.binomial(n=e_tested, p=p_TP)
            # undetected are those who are not tested + the FN tests
            # i.e. all but the true positives
            dn_e_iu[v,z] = leaving_E[v,z] - e_TP
            # 3) number sequenced
            e_sequenced = np.random.binomial(n=e_TP, p=p_gs[z])
            # 4.1) quarantine & sequenced
            dn_e_igq[v,z] = np.random.binomial(n=e_sequenced, p=p_q)
            dn_e_ignq[v,z] = e_sequenced - dn_e_igq[v,z]
            # quarantine & tested
            dn_e_itq[v,z] = np.random.binomial(n=e_TP-e_sequenced, p=p_q)
            dn_e_itnq[v,z] = e_tested - e_sequenced - dn_e_itq[v,z]

    ### ------- Flows out of I -------
    dn_iu_ru = np.zeros(shape=(n_variants, n_locations))
    dn_itq_rt = np.zeros(shape=(n_variants, n_locations))
    dn_itnq_rt = np.zeros(shape=(n_variants, n_locations))
    dn_igq_rg = np.zeros(shape=(n_variants, n_locations))
    dn_ignq_rg = np.zeros(shape=(n_variants, n_locations))

    for v in np.arange(n_variants):
        for z in np.arange(n_locations):
            dn_iu_ru[v,z] = np.random.binomial(n=int(Iu[v,z]), p=1/D[v])
            dn_itq_rt[v,z] = np.random.binomial(n=int(Itq[v,z]), p=1/D[v])
            dn_itnq_rt[v,z] = np.random.binomial(n=int(Itnq[v,z]), p=1/D[v])
            dn_igq_rg[v,z] = np.random.binomial(n=int(Igq[v,z]), p=1/D[v])
            dn_ignq_rg[v,z] = np.random.binomial(n=int(Ignq[v,z]), p=1/D[v])

    ### ------- Flows out of R -------
    dn_ru_sself = np.zeros(shape=(n_variants, n_locations))
    dn_ru_sx = np.zeros(shape=(n_variants, n_locations))
    dn_ru_sboth = np.zeros(shape=(n_variants, n_locations))
    dn_rt_sself = np.zeros(shape=(n_variants, n_locations))
    dn_rt_sx = np.zeros(shape=(n_variants, n_locations))
    dn_rt_sboth = np.zeros(shape=(n_variants, n_locations))
    dn_rg_sself = np.zeros(shape=(n_variants, n_locations))
    dn_rg_sx = np.zeros(shape=(n_variants, n_locations))
    dn_rg_sboth = np.zeros(shape=(n_variants, n_locations))

    for v in np.arange(n_variants):
        for z in np.arange(n_locations):
            # first leaving each R compartment
            leaving_Ru = np.random.binomial(n=int(Ru[v,z]), p=w_self[v]+w_x[v]+w_self[v]*w_x[v])
            leaving_Rt = np.random.binomial(n=int(Rt[v,z]), p=w_self[v]+w_x[v]+w_self[v]*w_x[v])
            leaving_Rg = np.random.binomial(n=int(Rg[v,z]), p=w_self[v]+w_x[v]+w_self[v]*w_x[v])
            # then divide across the 3 S compartments
            p_per_S_temp = np.array([w_self[v]*(1-w_x[v]), w_x[v]*(1-w_self[v]), w_self[v]*w_x[v]])
            p_per_S = np.divide(p_per_S_temp, np.sum(p_per_S_temp,axis=0))
            dn_ru_sself[v,z], dn_ru_sx[v,z], dn_ru_sboth[v,z] = np.random.multinomial(n=leaving_Ru, pvals=p_per_S)
            dn_rt_sself[v,z], dn_rt_sx[v,z], dn_rt_sboth[v,z] = np.random.multinomial(n=leaving_Rt, pvals=p_per_S)
            dn_rg_sself[v,z], dn_rg_sx[v,z], dn_rg_sboth[v,z] = np.random.multinomial(n=leaving_Rg, pvals=p_per_S)


    ### ------- Reinfections, flows out of re-S -------
    # the number of people being reinfected with the "same variant" in a given location
    dn_sself_eself = np.zeros(shape=(n_variants,n_locations))
    # the number of people who are susceptible to the same variant also becoming susceptible to the other variant
    dn_sself_sboth = np.zeros(shape=(n_variants,n_locations))
    # the number of people who are susceptible to the other variant also becoming susceptible to the own variant
    dn_sx_sboth = np.zeros(shape=(n_variants,n_locations))
    # self-reinfection from the sboth compartment
    dn_sboth_eself = np.zeros(shape=(n_variants,n_locations))
    # for the movement between variants, we distinguish between origin and destination
    # origin indicates the variant that is losing people
    # destination indicates the variants that are receiving people
    # origin is subtracted from the Sx, Sboth compartments
    # destination is added to the E compartments
    # origin and destination should have the same sum
    dn_sx_eother_destination = np.zeros(shape=(n_variants,n_locations))
    dn_sboth_eother_destination = np.zeros(shape=(n_variants,n_locations))
    dn_sx_eother_origin = np.zeros(shape=(n_variants,n_locations))
    dn_sboth_eother_origin = np.zeros(shape=(n_variants,n_locations))

    
    for v in np.arange(n_variants):
        for z in np.arange(n_locations):
            # Self-reinfection from Sself
            dn_sself_eself[v,z], dn_sself_sboth[v,z], fool_remaining = np.random.multinomial(n=int(S_self[v,z]), 
                                                                                             pvals=[a_self[v]*lambda_param[v,z],
                                                                                                    w_x[v],
                                                                                                    1-(w_x[v]+(a_self[v]*lambda_param[v,z]))
                                                                                                   ])

            other_variants = [x for x in np.arange(n_variants) if x != v]

            # probability of re-infection with each of the other variants
            # multiplied by reduced infectiousness factor            
            #p_inf_other = a_cross[v]*lambda_param[other_variants,z]
            
            p_leaving_sx = List()
            sum_p_leaving_sx = 0.
            for xx in np.arange(len(other_variants)):
                append_val = a_cross[v]*lambda_param[other_variants[xx],z]
                p_leaving_sx.append(append_val)
                sum_p_leaving_sx += append_val
            sum_p_leaving_sx_1 = sum_p_leaving_sx + w_self[v]

            # Cross-reinfection from Sx
            ## moving from sx to an e compartment or to sboth
            fool = np.random.multinomial(n=int(S_x[v,z]), 
                                        pvals=list(p_leaving_sx) + [w_self[v]] + [1-sum_p_leaving_sx_1])
            
            # had to make adjustments for numba implementation, hard coded 2 variants
            assert len(other_variants)<2, "Can currently only handle two variants"
            
            # add to the output matrix for reinfections
            # destination
            dn_sx_eother_destination[other_variants[0],z] += fool[0]
            # origin
            dn_sx_eother_origin[v,z] = np.sum(fool[:-2])
            # save in the output matrix for sx to sboth flow
            dn_sx_sboth[v,z] = fool[-2]
            
            # Cross- and self-reinfection from Sboth
            # probs:  moving from sboth to another e compartment, probability of self-reinfection, probabilitiy of staying
            sum_p_leaving_sx_2 = sum_p_leaving_sx + a_self[v]*lambda_param[v,z]
            fool = np.random.multinomial(n=int(S_both[v,z]), 
                                         pvals=list(p_leaving_sx) + [a_self[v]*lambda_param[v,z]] + [1-sum_p_leaving_sx_2])
            
            # add to the output matrix for reinfections
            # destination
            dn_sboth_eother_destination[other_variants[0],z] += fool[0]
            # origin
            dn_sboth_eother_origin[v,z] = np.sum(fool[:-2])
            # self reinfection
            dn_sboth_eself[v,z] = fool[-2]
            
            
    ## combine the flows to compute the next compartment sizes
    S_next = S + dn_sq_s - dn_s_sq - np.sum(dn_s_e,axis=0)
    Sq_next = Sq + dn_s_sq - dn_sq_s
    E_next = E + dn_s_e + dn_sx_eother_destination + dn_sboth_eother_destination + dn_sself_eself + dn_sboth_eself - leaving_E

    Iu_next = Iu + dn_e_iu - dn_iu_ru
    Itq_next = Itq + dn_e_itq - dn_itq_rt
    Itnq_next = Itnq + dn_e_itnq - dn_itnq_rt
    Igq_next = Igq + dn_e_igq - dn_igq_rg
    Ignq_next = Ignq + dn_e_ignq - dn_ignq_rg

    Ru_next = Ru + dn_iu_ru - dn_ru_sself - dn_ru_sx - dn_ru_sboth
    Rt_next = Rt + dn_itq_rt + dn_itnq_rt - dn_rt_sself - dn_rt_sx - dn_rt_sboth
    Rg_next = Rg + dn_igq_rg + dn_ignq_rg - dn_rg_sself - dn_rg_sx - dn_rg_sboth

    S_self_next = S_self + dn_ru_sself + dn_rt_sself + dn_rg_sself - dn_sself_eself - dn_sself_sboth
    S_x_next = S_x + dn_ru_sx + dn_rt_sx + dn_rg_sx - dn_sx_sboth - dn_sx_eother_origin
    S_both_next = S_both + (dn_ru_sboth + dn_rt_sboth + dn_rg_sboth) + (dn_sx_sboth + dn_sself_sboth) - dn_sboth_eself - dn_sboth_eother_origin
    
    next_compartments = (S_next, Sq_next, E_next, Iu_next, Itq_next, Itnq_next, Igq_next, Ignq_next, 
                         Ru_next, Rt_next, Rg_next, S_self_next, S_x_next, S_both_next)
    
    # separately save some flows to compute the cumulative exposures, infections, and recoveries
    next_cumulative = (dn_s_e, dn_sx_eother_destination, dn_sboth_eother_destination, dn_sself_eself, dn_sboth_eself, # for cumulative E
                       dn_e_iu, dn_e_itq, dn_e_itnq, dn_e_igq, dn_e_ignq, # for cumulative I's
                       dn_iu_ru, dn_itq_rt, dn_itnq_rt, dn_igq_rg, dn_ignq_rg # for cumulative R's
                      )
    
    return next_compartments, next_cumulative





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
            
            # update params:
            fb_kmatrix, b, D, L, Q, p_ta, p_ts, p_gs, p_TP, p_FP, p_q, theta, w_self, w_x, a_self, a_cross, n_variants, n_locations = params
            params = (fb_kmatrix, b, D, L, Q, p_ta, p_ts, p_gs_new, p_TP, p_FP, p_q, theta, w_self, w_x, a_self, a_cross, n_variants, n_locations)

                    
    return t_first_sequenced, patch_first_sequenced, e_cumulative, e_re_cumulative, iu_cumulative, it_cumulative, ig_cumulative, Iu_series, It_series, Ig_series
    








