

############################################################## 
# PART B : plots for extreme strategies
############################################################## 




import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



plot_folder = 'plots/'


extreme_proportions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
intro_times = [0, 50, 80, 100, 120, 150]
intro_locations = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten']

colors = sns.color_palette("hls", 8)


# ------------------------------------------------------------
# Read Data
# ------------------------------------------------------------

# read cumulative exposures
e_cumulative_all = np.load('extreme_strategies_intermediates/combo_e_cumulative_all.npy')
# shape: extreme proportion x extreme location x intro_locations x intro_times x n_variants x locations x simulation iterations


# read detection time
t_first_sequenced_all = np.load('extreme_strategies_intermediates/combo_t_first_sequenced_all.npy')


# replace zeros with nan
t_first_sequenced_all[t_first_sequenced_all==0.] = np.nan

### subtract intro time from sequencing time
# only for second variant
t_first_sequenced_all[:,:,:,:,1,:,] = np.subtract(t_first_sequenced_all[:,:,:,:,1,:], 
                                                        np.array(intro_times)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])


# resulting shape: (extreme proportion x extreme location x intro_locations x intro_times x n_variants)
detection_t_mean = np.nanmean(t_first_sequenced_all, axis=5)  # shape ()








# ------------------------------------------------------------
# Figure 2
# ------------------------------------------------------------



ncols=1
nrows=1

fig, axs = plt.subplots(nrows, ncols, figsize=(7*ncols, 7*nrows))

# prep the two datasets
use_data_same = pd.DataFrame()
use_data_different = pd.DataFrame()

for tt, intro_time in enumerate(intro_times):
    
    for ll, intro_loc in enumerate(intro_locations):
        
        for ii, extreme_loc in enumerate(intro_locations): 
            
            for ee, extreme_prop in enumerate(extreme_proportions): 
                
                if ii == ll:
                    df_temp1 = pd.DataFrame({'detection time' : t_first_sequenced_all[ee,ii,ll,tt,1,:].flatten()})
                    df_temp1['extreme_prop'] = extreme_prop
                    
                    use_data_same = use_data_same.append(df_temp1)
                    
                if ii != ll:
                    df_temp2 = pd.DataFrame({'detection time' : t_first_sequenced_all[ee,ii,ll,tt,1,:].flatten()})
                    df_temp2['extreme_prop'] = extreme_prop
                    
                    use_data_different = use_data_different.append(df_temp2)

sns.lineplot(x='extreme_prop', y='detection time', 
             data=use_data_same, color=colors[5], 
             ax=axs, label='in the same location')

sns.lineplot(x='extreme_prop', y='detection time', 
             data=use_data_different, color=colors[0], 
             ax=axs, label='in different locations')




axs.set_xlabel('Proportion of tests \n allocated in single borough', fontsize=20)
axs.set_ylabel('Detection time \n (days)', fontsize=20)

axs.tick_params(axis='x', which='major', labelsize=20, width=0, length=0)
axs.tick_params(axis='y', which='major', labelsize=20, width=3, length=5)


# remove axes
axs.spines.top.set_visible(False)
axs.spines.right.set_visible(False)
# spine width
axs.spines.left.set_linewidth(3)
axs.spines.bottom.set_linewidth(3)

axs.legend(title='over-sampling and \nnew variant emergence',
           fontsize=15, title_fontsize=15, 
           )

plt.tight_layout()

plt.savefig(plot_folder + 'figure2.pdf', dpi=300)







