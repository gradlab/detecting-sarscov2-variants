
############################################################## 
# PART A: prep data for viz & analysis: combine sim results 
############################################################## 



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




# set location of individual files
out_folder_preamble = 'sensitivity_analysis_varying_incubation/'
# plot folder
plot_folder = 'sensitivity_analysis_varying_incubation/figures/'

# set the dimensions of the individual data files, based on len of parameter values
latent_durations = [3, 5, 7]
intro_locations = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten']
intro_times = [0, 50, 80, 100, 120, 150]

n_latent = len(latent_durations)
n_variants = 2
n_iter = 100


# set the outputs
e_cumulative_all = np.zeros(shape=( n_latent, len(intro_locations), len(intro_times),
                                   n_variants, len(intro_locations), n_iter) )
t_first_sequenced_all = np.zeros(shape=( n_latent, len(intro_locations), len(intro_times),
                                   n_variants, n_iter) )


for l_iter in np.arange(len(latent_durations)):

    l_val = latent_durations[l_iter]
    L = [5,l_val]  # latent period

    out_folder = out_folder_preamble+'lvalue_'+str(l_iter)+'/results_density/'
    
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
    

    e_cumulative_all[l_iter,:,:,:,:,:] = e_temp
    t_first_sequenced_all[l_iter,:,:,:,:] = t_first_sequenced_temp
    

# save
np.save(out_folder_preamble+'combo_e_cumulative_all.npy', e_cumulative_all)
np.save(out_folder_preamble+'combo_t_first_sequenced_all.npy', t_first_sequenced_all)






############################################################## 
# PART B : plots comparing average contacts
############################################################## 





daily_tests = 7183


colors = sns.color_palette("hls", 8)

# to scale test quantity by 100k pop
N_multiplier_100k = 8253213/100000 #population in 100k
# divide the tests by the multiplier to get the quantity per 100k pop



t_first_sequenced_all[t_first_sequenced_all==0.] = np.nan


### subtract intro time from sequencing time
# only for second variant
t_first_sequenced_all[:,:,:,1,:] = np.subtract(t_first_sequenced_all[:,:,:,1,:], 
                                                        np.array(intro_times)[np.newaxis, np.newaxis, :, np.newaxis])





# ---------------------------------------------------------------------
# BOXPLOTS
# ---------------------------------------------------------------------

# note: since contact rate impacts disease dynamics it only makes sense to 
# compare introduction time 0

tt=0

# --- boxplot 1: detection time


# flatten detection time data for swarmplot 
use_data = pd.DataFrame()


for l_iter in np.arange(len(latent_durations)):
    l_val = latent_durations[l_iter]    
    df_temp = pd.DataFrame({'detection time' : t_first_sequenced_all[l_iter,:,tt,1,:].flatten()})
    df_temp['l_val'] = str(l_val)
    
    use_data = use_data.append(df_temp)

use_data.reset_index(inplace=True, drop=True)

# **** make swarmplot ********

fig, axs = plt.subplots(1,1, figsize=(14,10))

sns.swarmplot(x='l_val', y='detection time', data=use_data, 
                hue='l_val', alpha=0.5, s=6,
                ax=axs, 
                #legend=False, 
                palette=[colors[5], colors[0], colors[2]],
                #zorder=1
                )

sns.boxplot(data=use_data,
            y='detection time',
            x='l_val',
            color = 'white',
            width = 0.5,
            ax=axs,
            linewidth = 3,
            showfliers = False)

for i,artist in enumerate(axs.artists):
    boxlinecolor = 'black'
    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    artist.set_edgecolor(boxlinecolor)

    # Each box has 5 associated objects (to make the whiskers, fliers, etc.)
    for j in range(i*5,i*5+5):
        line = axs.lines[j]
        line.set_color(boxlinecolor)
        line.set_mfc(boxlinecolor)
        line.set_mec(boxlinecolor)

axs.set_xlabel('Latent period of new variant \n(days)', fontsize=20)
axs.set_ylabel('Detection time \n (days)', fontsize=20)
axs.set_xticklabels(['3', '5', '7'])

axs.tick_params(axis='x', which='major', labelsize=20, width=0, length=0)
axs.tick_params(axis='y', which='major', labelsize=20, width=3, length=5)
  
axs.legend_.remove()
    
   
# remove axes
axs.spines.top.set_visible(False)
#axs.spines.bottom.set_visible(False)
#axs.spines.left.set_visible(False)
axs.spines.right.set_visible(False)
# spine width
axs.spines.left.set_linewidth(3)
axs.spines.bottom.set_linewidth(3)

plt.savefig(plot_folder+'boxplot-detection-time_compare-l-vals_intro-time-'+str(tt)+'.pdf', dpi=300)
    

# --- boxplot 2: burden



# flatten detection time data for swarmplot 
use_data = pd.DataFrame()


for l_iter in np.arange(len(latent_durations)):
    l_val = latent_durations[l_iter]
    df_temp = pd.DataFrame({'cumulative infections' : np.sum(e_cumulative_all[l_iter,:,tt,1,:,:], axis=1).flatten()})
    df_temp['l_val'] = str(l_val)
    
    use_data = use_data.append(df_temp)

use_data.reset_index(inplace=True, drop=True)

# **** make swarmplot ********

fig, axs = plt.subplots(1,1, figsize=(14,10))

sns.swarmplot(x='l_val', y='cumulative infections', data=use_data, 
                hue='l_val', alpha=0.5, s=6,
                ax=axs, 
                #legend=False, 
                palette=[colors[5], colors[0], colors[2]],
                #zorder=1
                )

sns.boxplot(data=use_data,
            y='cumulative infections',
            x='l_val',
            color = 'white',
            width = 0.5,
            ax=axs,
            linewidth = 3,
            showfliers = False)

for i,artist in enumerate(axs.artists):
    boxlinecolor = 'black'
    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    artist.set_edgecolor(boxlinecolor)

    # Each box has 5 associated objects (to make the whiskers, fliers, etc.)
    for j in range(i*5,i*5+5):
        line = axs.lines[j]
        line.set_color(boxlinecolor)
        line.set_mfc(boxlinecolor)
        line.set_mec(boxlinecolor)

axs.set_xlabel('Latent period of new variant \n(days)', fontsize=20)
axs.set_ylabel('Cumulative infections', fontsize=20)
axs.set_xticklabels(['3', '5', '7'])

axs.tick_params(axis='x', which='major', labelsize=20, width=0, length=0)
axs.tick_params(axis='y', which='major', labelsize=20, width=3, length=5)
  
axs.legend_.remove()
    
   
# remove axes
axs.spines.top.set_visible(False)
#axs.spines.bottom.set_visible(False)
#axs.spines.left.set_visible(False)
axs.spines.right.set_visible(False)
# spine width
axs.spines.left.set_linewidth(3)
axs.spines.bottom.set_linewidth(3)

plt.savefig(plot_folder+'boxplot-cumulative-infections_compare-l-vals_intro-time-'+str(tt)+'.pdf', dpi=300)


