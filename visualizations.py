

############################################################## 
# PART A : plots comparing testing and sequencing rates
############################################################## 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap



plot_folder = 'plots/'

strategies = ['statusquo', 'density', 'random']
seqrates = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
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

daily_tests = 7183


colors = sns.color_palette("hls", 8)



# to scale test quantity by 100k pop
N_multiplier_100k = 8253213/100000 #population in 100k
# divide the tests by the multiplier to get the quantity per 100k pop



# functions

def nested_shapes(data, labels=None, c=None, ax=None, 
                   cmap=None, norm=None, textkw={}):
    ax = ax or plt.gca()
    data = np.array(data)
    R = np.sqrt(data/data.max())
    # circle:
    #p = [plt.Circle((0,r), radius=r) for r in R[::-1]]
    # square:
    p = [plt.Rectangle(xy=(0,0), width=r, height=r) for r in R[::-1]]
    arr = data[::-1] if c is None else np.array(c[::-1])
    col = PatchCollection(p, cmap=cmap, norm=norm, array=arr)

    ax.add_collection(col)
    ax.axis("off")
    ax.set_aspect("equal")
    ax.autoscale()

    if labels is not None:
        kw = dict(color="white", va="bottom", ha="left")
        kw.update(textkw)
        ax.text(0, R[0], labels[0], **kw)
        for i in range(1, len(R)):
            ax.text(0, R[i]+R[i-1], labels[i], **kw)
    return col




# ------------------------------------------------------------
# Read Data
# ------------------------------------------------------------

# read cumulative exposures
# shape: 
e_cumulative_all = np.load('transmission_rate_multiples/combo_e_cumulative_all.npy')


# read detection time
t_first_sequenced_all = np.load('transmission_rate_multiples/combo_t_first_sequenced_all.npy')
# replace zeros with nan
t_first_sequenced_all[t_first_sequenced_all==0.] = np.nan


### subtract intro time from sequencing time
# only for second variant
t_first_sequenced_all[:,:,:,:,:,:,1,:] = np.subtract(t_first_sequenced_all[:,:,:,:,:,:,1,:], 
                                                        np.array(intro_times)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])






# ---------------------------------------------------------------------
# FIGURE 1 
# ---------------------------------------------------------------------

# define scenario
bb = 2
tt = 1
mm = 3
ss = 1


# flatten detection time data for swarmplot 
use_data = pd.DataFrame()

for ff,stratname in enumerate(strategies):
    df_temp = pd.DataFrame({'detection time' : t_first_sequenced_all[bb,ff,mm,ss,:,tt,1,:].flatten()})
    df_temp['strategy'] = stratname
    # add xs with jitter for the scatter plot
    df_temp['xs'] = np.random.normal(ff, 0.1, df_temp.shape[0])
    
    use_data = use_data.append(df_temp)


# **** make swarmplot ********

fig, axs = plt.subplots(1,1, figsize=(14,10))

sns.swarmplot(x='strategy', y='detection time', data=use_data, 
                hue='strategy', alpha=0.5, s=6,
                ax=axs, 
                #legend=False, 
                palette=[colors[5], colors[0], colors[2]],
                zorder=1
                )

sns.boxplot(data=use_data,
            y='detection time',
            x='strategy',
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

axs.set_xlabel('')
axs.set_ylabel('Detection time \n (days)', fontsize=20)
axs.set_xticklabels(['Status quo', 'Population density', 'Random'])

axs.tick_params(axis='x', which='major', labelsize=20, width=0, length=0)
axs.tick_params(axis='y', which='major', labelsize=20, width=3, length=5)
  
axs.legend_.remove()
    
   
# remove axes
axs.spines.top.set_visible(False)
axs.spines.bottom.set_visible(False)
#axs.spines.left.set_visible(False)
axs.spines.right.set_visible(False)
# spine width
axs.spines.left.set_linewidth(3)

plt.savefig(plot_folder+'figure1.pdf', dpi=300)









# ---------------------------------------------------------------------
# FIGURE 3 A
# ---------------------------------------------------------------------


ff = 0
bb = 2
tt = 1
intro_time=50


fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

for ss, seqrate in enumerate(seqrates):
    
    # prep data for lineplot
    num_seqs_all = []
    num_tests_all = []
    det_time_all = []
    
    # compute the number of tests and sequences
    num_seqs = [x * daily_tests * seqrate for x in multipliers]
    num_tests = [x * daily_tests for x in multipliers]
    
    for i in np.arange(t_first_sequenced_all.shape[7]):
        # correponding detection time (average):
        det_time = np.mean(t_first_sequenced_all[bb,ff,:,ss,:,tt,1,i], axis=(1))
        
        # append
        num_seqs_all.extend(num_seqs)
        num_tests_all.extend(num_tests)
        det_time_all.extend(det_time)
        
    num_tests_all_100k = [x/N_multiplier_100k for x in num_tests_all]

    sns.lineplot(y=det_time_all, 
                 x=num_tests_all_100k,
                 color = colors[ss],
                 label = str(seqrate),
                 ax=axs,
                 linewidth=5
                 )
axs.set_xlabel('Test volume \n (per 100k persons per day)', fontsize=25)
axs.set_ylabel('Detection time \n (days)', fontsize=25)
axs.legend(title='Sequencing rate', fontsize = 20, title_fontsize=25)
# tick sizes
axs.tick_params(axis='both', which='major', labelsize=25, width=3, length=5)


# format x label
current_values = plt.gca().get_xticks()
plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_values])

# set the axes size
axs.spines.left.set_linewidth(3)
axs.spines.bottom.set_linewidth(3) 
# remove axes
axs.spines.top.set_visible(False)
axs.spines.right.set_visible(False)
      
                
plt.tight_layout()
plt.savefig(plot_folder+'figure3a.pdf', dpi=300)










# ---------------------------------------------------------------------
# FIGURE 3 B
# ---------------------------------------------------------------------



ff = 0
tt = 1
bb = 2


fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

for ss, seqrate in enumerate(seqrates):
    
    # prep data for lineplot
    num_seqs_all = []
    num_tests_all = []
    e_cum_temp = []
    
    # compute the number of tests and sequences
    num_seqs = [x * daily_tests * seqrate for x in multipliers]
    num_tests = [x * daily_tests for x in multipliers]
    
    for i in np.arange(e_cumulative_all.shape[8]):
        # correponding detection time (average):
        e_cum = np.mean(np.sum(e_cumulative_all[bb,ff,:,ss,:,tt,1,:,i], axis=2), axis=(1))
        
        # append
        num_seqs_all.extend(num_seqs)
        num_tests_all.extend(num_tests)
        e_cum_temp.extend(e_cum)
    
    num_tests_all_100k = [x/N_multiplier_100k for x in num_tests_all]
    e_cum_temp_100k = [x/N_multiplier_100k for x in e_cum_temp]
    
    sns.lineplot(y=e_cum_temp_100k, 
                 x=num_tests_all_100k,
                 color = colors[ss],
                 label = str(seqrate),
                 ax=axs,
                 linewidth=5
                 )
axs.set_xlabel('Test volume \n (per 100k persons per day)', fontsize=25)
axs.set_ylabel('Cumulative infections \n (per 100k persons)', fontsize=25)
axs.legend(title='Sequencing rate', fontsize = 20, title_fontsize=25)
# tick sizes
axs.tick_params(axis='both', which='major', labelsize=25, width=3, length=5)


# format x label
current_values = plt.gca().get_xticks()
plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_values])

# set the axes size
axs.spines.left.set_linewidth(3)
axs.spines.bottom.set_linewidth(3) 
# remove axes
axs.spines.top.set_visible(False)
axs.spines.right.set_visible(False)
      
                
plt.tight_layout()
plt.savefig(plot_folder+'figure3b.pdf', dpi=300)











# ---------------------------------------------------------------------
# FIGURE 5 A
# ---------------------------------------------------------------------





ff = 0
mm = 3
ss = 1
bb = 2

# make data for scatter and box plots

use_data = pd.DataFrame()
for tt, intro_time in enumerate(intro_times):
    df_temp = pd.DataFrame({'detection time' : t_first_sequenced_all[bb,ff,mm,ss,:,tt,1,:].flatten()})
    df_temp['intro time'] = intro_time
    # add xs with jitter for the scatter plot
    df_temp['xs'] = np.random.normal(tt, 0.1, df_temp.shape[0])
    
    use_data = use_data.append(df_temp)


fig, axs = plt.subplots(1, 1, figsize=(14,10))


# alternative: boxplot instead of 95% CI
sns.scatterplot(x='xs', y='detection time', data=use_data, 
                #hue='intro time', 
                alpha=0.3, s=50,
                ax=axs, legend=None, color = colors[0],
                zorder=1)

sns.boxplot(data=use_data,
            y='detection time',
            x='intro time',
            color = 'white',
            width = 0.5,
            ax=axs,
            linewidth = 3)

for i,artist in enumerate(axs.artists):
    boxlinecolor = 'black'
    # Set the linecolor on the artist to the facecolor, and set the facecolor to None
    artist.set_edgecolor(boxlinecolor)

    # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
    for j in range(i*6,i*6+6):
        line = axs.lines[j]
        line.set_color(boxlinecolor)
        line.set_mfc(boxlinecolor)
        line.set_mec(boxlinecolor)

axs.set_xlabel('Introduction time of second variant \n (days since start of simulation)', fontsize=20)
axs.set_ylabel('Detection time \n (days)', fontsize=20)

axs.tick_params(axis='x', which='major', labelsize=20, width=0, length=0)
axs.tick_params(axis='y', which='major', labelsize=20, width=3, length=5)
      
   
# remove axes
axs.spines.top.set_visible(False)
axs.spines.right.set_visible(False)
# spine width
axs.spines.left.set_linewidth(3)
axs.spines.bottom.set_linewidth(3)


plt.savefig(plot_folder+'figure5a.pdf', 
            dpi=300)






# ---------------------------------------------------------------------
# FIGURE 5 B
# ---------------------------------------------------------------------


# share taken over the simulation iterations & intro locations
extinction_prob = np.mean( ( np.isnan(t_first_sequenced_all[:,:,:,:,:,:,1,:]) ).astype(int), axis=(4,6) )



ff = 0
mm = 3
ss = 1
bb = 2

        
nrows = 1
ncols = 6
fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows), 
                sharex=True, sharey=True)

Ext_vals = extinction_prob[bb,ff,mm,ss,:]

for v,val in enumerate(Ext_vals):
    
    # determine position of label
    if val < 0.1:
        position = (0.,0.)
    else:
        position = (0.5*val, 0.4*val)
    nested_shapes([val,1], labels=[str( round(val*100, 1))+'%', ''], 
                   cmap = ListedColormap([colors[0], 
                                          (0.8274509803921568, 0.8274509803921568, 0.8274509803921568)]),
                   textkw=dict(fontsize=45, position=(0.4,0.4), color='black'),
                   ax=axs[v])
plt.suptitle('Extinction probability', fontsize=45)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])

plt.savefig(plot_folder+'figure5b.pdf', dpi=300, )










# ---------------------------------------------------------------------
# FIGURE 6
# ---------------------------------------------------------------------


ff = 0
mm = 3
ss = 1
tt = 1
bb = 2

fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10,9))
        
for bb, bval in enumerate(bvalues):
            
    t_slice = np.mean(t_first_sequenced_all[bb,ff,mm,ss,:,tt,1,:],
                      axis=0)
    # cumulative e, summed over all boroughs
    e_slice = np.sum(np.nanmean(e_cumulative_all[bb,ff,mm,ss,:,tt,1,:,:],
                             axis=0),
                     axis=0)
        
    e_slice_100k = [x/N_multiplier_100k for x in e_slice]

    sns.scatterplot(x = e_slice_100k, y = t_slice, 
                    color=colors[bb], alpha=0.7, s=130,
                    label=str(bval))
    
    
axs.set_ylabel('Detection time \n (days)', fontsize=14)
axs.set_xlabel('Cumulative infections (per 100k persons)', fontsize=14)
# legend
lgd = axs.legend(title='Transmission rate', loc='best', bbox_to_anchor=(1., 1.), 
                 fontsize=12, title_fontsize=14)
# format y axis labels
current_values = plt.gca().get_xticks()
plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_values], rotation=45, fontsize=12)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(current_values.astype(int), fontsize=12)
axs.spines.top.set_visible(False)
axs.spines.right.set_visible(False)

axs.spines.left.set_linewidth(2)
axs.spines.bottom.set_linewidth(2)

axs.tick_params(axis='x', which='major',  width=2, length=4)
axs.tick_params(axis='y', which='major', width=2, length=4)

plt.savefig(plot_folder+'figure6.pdf', 
            dpi=300,
            bbox_extra_artists=(lgd,), bbox_inches='tight')
















