
############################################################## 
# PART C : plots for fixed resources
############################################################## 




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import variants_parameters_more_transmissible as variant_pars



# colors
colors = sns.color_palette("hls", 19)


plot_folder = 'plots/'




quantity_tests = [500, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10500, 12500, 14500, 16500]
quantity_sequences = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4500, 5500, 6500, 7500, 9000, 10500, 12000, 13500, 15000]

                
intro_times = [0, 50, 80, 100, 120, 150]



n_strategies = 1
n_multipliers = len(quantity_tests)
n_seqrates = len(quantity_sequences)
n_bvalues = 1
n_variants = 2
n_iter = 100





#t_first_sequenced_all = np.load('fixed_resources_v2/combo_t_first_sequenced_all.npy')
t_first_sequenced_all = np.load('fixed_resources_v1/combo_t_first_sequenced_all.npy')

#e_cumulative_all = np.load('fixed_resources_v2/combo_e_cumulative_all.npy')
e_cumulative_all = np.load('fixed_resources_v1/combo_e_cumulative_all.npy')




test_rates = [quantity_tests/np.sum(variant_pars.N)]
seq_rates = [quantity_sequences[i]/quantity_tests[j] for i in np.arange(len(quantity_sequences)) for j in np.arange(len(quantity_tests))]



# to scale test quantity by 100k pop
N_multiplier_100k = 8253213/100000 #population in 100k




# ----------------------------------------------------------------------------------
# aggregate the output over the simulation iterations
# ----------------------------------------------------------------------------------

# replace zeros with nan's
t_first_sequenced_all[t_first_sequenced_all==0.] = np.nan

### subtract intro time from sequencing time
# only for second variant
t_first_sequenced_all[:,:,:,:,1,:] = np.subtract(t_first_sequenced_all[:,:,:,:,1,:], 
                                                        np.array(intro_times)[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])


# mean time to detection over simulation iterations (axis=5)
detection_t_mean = np.nanmean(t_first_sequenced_all, axis=5)  # shape ()





# ------------------------------------------------------------
# Figure 4 A
# ------------------------------------------------------------



quantity_tests_100k = [int(x/N_multiplier_100k) for x in quantity_tests]
quantity_sequences_100k = [int(x/N_multiplier_100k) for x in quantity_sequences]


tt = 1

fig, axs = plt.subplots(nrows=1, ncols=1, 
                        figsize=(15, 10))
# value is average across all intro locations
for ss, seq_quantity in enumerate(quantity_sequences_100k):
    sns.lineplot(y=np.mean( detection_t_mean[:,ss,:,tt,1], axis=(1)), 
                 x=quantity_tests_100k,
                 color=colors[ss],
                 label=str(seq_quantity),
                 ax=axs,
                 linewidth=5)
       
axs.set_ylabel('Detection time \n (days)',
               fontsize=20)
axs.set_xlabel('Test volume \n (per 100k persons per day)',
               fontsize=20)
lgd = axs.legend(title='Max. sequencing \nvolume \n(per 100k persons \nper day)', fontsize=15, title_fontsize=20, 
           bbox_to_anchor=(1., 1.))
# set the axes size
axs.spines.left.set_linewidth(3)
axs.spines.bottom.set_linewidth(3) 
# remove axes
axs.spines.top.set_visible(False)
axs.spines.right.set_visible(False)
# ticks
axs.tick_params(axis='both', which='major', labelsize=20, width=3, length=5)

plt.tight_layout()

plt.savefig(plot_folder+'figure4a.pdf', 
            dpi=300,
            bbox_extra_artists=(lgd,), bbox_inches='tight')








# ------------------------------------------------------------
# Figure 4 B
# ------------------------------------------------------------

quantity_tests_100k = [int(x/N_multiplier_100k) for x in quantity_tests]
quantity_sequences_100k = [int(x/N_multiplier_100k) for x in quantity_sequences]


tt = 1

fig, axs = plt.subplots(nrows=1, ncols=1, 
                        figsize=(15, 10))

for mm, test_quantity in enumerate(quantity_tests_100k):
    if mm>0:
        sns.lineplot(y=np.mean( detection_t_mean[mm,:,:,tt,1], axis=(1)), 
                     x=quantity_sequences_100k,
                     color=colors[mm],
                     label=str(test_quantity),
                     ax=axs,
                     linewidth=5)
        
axs.set_ylabel('Detection time \n (days)',
               fontsize=20)
axs.set_xlabel('Max. sequencing volume \n (per 100k persons per day)',
               fontsize=20)

lgd = axs.legend(title='Test volume \n (per 100k persons per day)', fontsize=15, title_fontsize=20, 
           bbox_to_anchor=(1., 1.))
# set the axes size
axs.spines.left.set_linewidth(3)
axs.spines.bottom.set_linewidth(3) 
# remove axes
axs.spines.top.set_visible(False)
axs.spines.right.set_visible(False)
# ticks
axs.tick_params(axis='both', which='major', labelsize=20, width=3, length=5)

plt.tight_layout()

plt.savefig(plot_folder+'figure4b.pdf', 
            dpi=300,
            bbox_extra_artists=(lgd,), bbox_inches='tight')








