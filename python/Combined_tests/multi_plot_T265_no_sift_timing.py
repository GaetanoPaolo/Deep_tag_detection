from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import h5py
f = h5py.File('/home/gaetan/data/hdf5/T265_alt_DBSCAN_8repr_clust_solve_timing_detect_hom_alt_debug_4inlier.hdf5', 'r+')
base_items = list(f.items())
dset2 = f.get(base_items[0][0])
trans_est_orb = np.array(dset2.get('trans_est_orb'))
drone_est = np.array(dset2.get('drone_est'))
tot_timing = np.array(dset2.get('timing'))

#computing errors
#orb_err = np.squeeze(abs(np.subtract(np.squeeze(drone_est, axis = 3),trans_est_orb[:,0:3])),axis = 2)
orb_err = np.squeeze(abs(np.subtract(np.squeeze(drone_est, axis = 3),trans_est_orb[:,0:3,:])),axis = 2)
#sift_err = np.squeeze(abs(np.subtract(np.squeeze(drone_est, axis = 3),trans_est_sift[:,0:3])),axis = 2)
drone_est = np.squeeze(np.squeeze(drone_est, axis = 3),axis = 2)
#separate the errors from the samples that couldn't be detected
orb_false_err = np.zeros((orb_err.shape[0],3))
orb_true_err = np.zeros((orb_err.shape[0],3))
print(trans_est_orb.shape)
print("ORB trans 280")
print(trans_est_orb[150,:])
print("GT trans 280")
print(drone_est[150,:])
#keeping track of values to delete
del_orb = []
del_sift = []
for j in range(0,orb_err.shape[0]):
    if np.sum(trans_est_orb[j,0:3,:]) == 0.0:
        #orb_false_err[j,:] = orb_err[j,:]
        orb_false_err[j,:] = np.nan
        orb_true_err[j,:] = np.nan
        del_orb.append(j)
    else:
        orb_true_err[j,:] = orb_err[j,:]
        orb_false_err[j,:] = np.nan



#setting the upper error bound that will be plotted
upper = 6
#setting up the begin timestamp of the simulation
begin = 100
size = drone_est.shape
end = size[0]
#plot each axis along altitude
fig, axd = plt.subplot_mosaic([#['zero'],
                                ['first'],
                               ['second'],
                               ['third'],
                               ['fourth']])
#l0, = axd['zero'].plot(drone_est[:,2],'.')
#axd['zero'].set_title('SIFT best resolution percentage')
l1, = axd['first'].plot(drone_est[begin:end,2],trans_est_orb[begin:end,3],'.')
axd['first'].set_title('ORB best resolution percentage')
axd['second'].set_title('Errors x-axis over altitude')
l2, = axd['second'].plot(drone_est[begin:end,2],orb_true_err[begin:end,0], '.',label = 'ORB')
axd['second'].set_ylim([0,2])
axd['second'].legend([l2],['ORB','SIFT'])
axd['third'].set_title('Errors y-axis over altitude')
l6,= axd['third'].plot(drone_est[begin:end,2],orb_true_err[begin:end,1], '.',label = 'ORB')
axd['third'].legend([l6],['ORB','SIFT'])
axd['third'].set_ylim([0,2])
axd['fourth'].set_title('Errors z-axis over altitude')
l10, = axd['fourth'].plot(drone_est[begin:end,2],orb_true_err[begin:end,2], '.',label = 'ORB')
axd['fourth'].legend([l10],['ORB','SIFT'])
axd['fourth'].set_ylim([0,2])
fig.suptitle('Estimation error for ORB and SIFT: ORB_create(2000,1.1,8,21,0,2,0,21,20)',fontsize=16)
plt.show()


#plotting error in each axis over time
fig, axd = plt.subplot_mosaic([#['first'],
                               ['second'],
                               ['third'],
                               ['fourth']])
# l1, = axd['first'].plot(range(0,size[0]),trans_est_orb[:,3],'.')
# axd['first'].set_title('ORB best resolution percentage')
axd['second'].set_title('Errors x-axis over timesamples')
l2, = axd['second'].plot(range(begin,size[0]),orb_true_err[begin:end,0],'.',label = 'ORB')
axd['second'].set_ylim([0,2])
axd['second'].legend([l2],['ORB'])
axd['third'].set_title('Errors y-axis over timesamples')
l4,= axd['third'].plot(range(begin,size[0]),orb_true_err[begin:end,1],'.',label = 'ORB')
axd['third'].legend([l4],['ORB','SIFT'])
axd['third'].set_ylim([0,2])
axd['fourth'].set_title('Errors z-axis over timesamples')
l6, = axd['fourth'].plot(range(begin,size[0]),orb_true_err[begin:end,2],'.',label = 'ORB')
axd['fourth'].legend([l6],['ORB'])
axd['fourth'].set_ylim([0,2])
#plt.ylim([0,20])
fig.suptitle('Estimation error for ORB and SIFT: ORB_create(2000,1.1,8,21,0,2,0,21,20)',fontsize=16)
plt.show()

#plot the estimated and true values for each axis
fig, axd = plt.subplot_mosaic([['first'],
                               ['second'],
                               ['third'],
                               ['fourth']])
l1, = axd['first'].plot(range(begin,size[0]),tot_timing[begin:end],'.')
axd['first'].set_ylim([0,0.5])
axd['first'].set_title('Algorithm timing')
axd['second'].set_title('x-axis tag pos relative to camera over timesamples')
l2, = axd['second'].plot(range(begin,size[0]),trans_est_orb[begin:end,0],'.',label = 'ORB')
l4, = axd['second'].plot(range(begin,size[0]),drone_est[begin:end,0],label = 'GT')
axd['second'].set_ylim([-3,5])
#axd['second'].set_xlim([400,600])
axd['second'].legend([l2,l4],['ORB','GT'])
axd['third'].set_title('Errors y-axis over timesamples')
l5,= axd['third'].plot(range(begin,size[0]),trans_est_orb[begin:end,1],'.',label = 'ORB')
l7, = axd['third'].plot(range(begin,size[0]),drone_est[begin:end,1],label = 'GT')
axd['third'].legend([l5,l7],['ORB','GT'])
axd['third'].set_ylim([-3,5])
#axd['third'].set_xlim([400,600])
axd['fourth'].set_title('Errors z-axis over timesamples')
l8, = axd['fourth'].plot(range(begin,size[0]),trans_est_orb[begin:end,2],'.',label = 'ORB')
l10, = axd['fourth'].plot(range(begin,size[0]),drone_est[begin:end,2],label = 'GT')
axd['fourth'].legend([l8,l10],['ORB','GT'])
axd['fourth'].set_ylim([0,10])
#axd['fourth'].set_xlim([400,600])
fig.suptitle('Estimation position for ORB and SIFT: ORB_create(2000,1.1,8,21,0,2,0,21,20)',fontsize=16)
plt.show()

#plot the estimated and true values for each axis zoom
fig, axd = plt.subplot_mosaic([['first'],
                               ['second'],
                               ['third'],
                               ['fourth']])
l1, = axd['first'].plot(range(begin,size[0]),tot_timing[begin:end],'.')
axd['first'].set_ylim([0,0.5])
axd['first'].set_xlim([100,200])
axd['first'].set_title('Algorithm timing')
axd['second'].set_title('x-axis tag pos relative to camera over timesamples')
l2, = axd['second'].plot(range(begin,size[0]),trans_est_orb[begin:end,0],'.',label = 'ORB')
l4, = axd['second'].plot(range(begin,size[0]),drone_est[begin:end,0],label = 'GT')
axd['second'].set_ylim([-1,1])
axd['second'].set_xlim([100,200])
#axd['second'].set_xlim([400,600])
axd['second'].legend([l2,l4],['ORB','GT'])
axd['third'].set_title('Errors y-axis over timesamples')
l5,= axd['third'].plot(range(begin,size[0]),trans_est_orb[begin:end,1],'.',label = 'ORB')
l7, = axd['third'].plot(range(begin,size[0]),drone_est[begin:end,1],label = 'GT')
axd['third'].legend([l5,l7],['ORB','GT'])
axd['third'].set_ylim([-1,1])
axd['third'].set_xlim([100,200])
#axd['third'].set_xlim([400,600])
axd['fourth'].set_title('Errors z-axis over timesamples')
l8, = axd['fourth'].plot(range(begin,size[0]),trans_est_orb[begin:end,2],'.',label = 'ORB')
l10, = axd['fourth'].plot(range(begin,size[0]),drone_est[begin:end,2],label = 'GT')
axd['fourth'].legend([l8,l10],['ORB','GT'])
axd['fourth'].set_ylim([0,2])
axd['fourth'].set_xlim([100,200])
#axd['fourth'].set_xlim([400,600])
fig.suptitle('Estimation position for ORB and SIFT: ORB_create(2000,1.1,8,21,0,2,0,21,20)',fontsize=16)
plt.show()

#calculating mean error for each component
orb_pruned = np.delete(orb_true_err,del_orb,axis = 0)
#plot the percentages of valid and invalid samples
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Valid', 'non-valid'
sizes = [len(orb_pruned), len(del_orb)-begin]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title('Ratios of valid and invalid samples (due to inlier threshold = 20)')
plt.show()



# mae_orb = np.mean(orb_pruned, axis = 0)
# print('MAE ORB')
# print(mae_orb)
# print('over')
# print(orb_pruned.shape)

#classify each error in categories:
#a)err < 10cm
#b) 10cm < err < 25cm
#c)25 cm < err < 50cm
#d) 50cm < err
a = [0,0,0]
b = [0,0,0]
c = [0,0,0]
d = [0,0,0]
for j in range(0,3):
    for i in range(0,len(orb_pruned)):
        if orb_pruned[i][j] < 0.1:
            a[j] += 1
        elif orb_pruned[i][j] < 0.25:
            b[j] += 1
        elif orb_pruned[i][j] < 0.5:
            c[j] += 1
        else:
            d[j] += 1
    a[j] = a[j]/len(orb_pruned)*100
    b[j] = b[j]/len(orb_pruned)*100
    c[j] = c[j]/len(orb_pruned)*100
    d[j] = d[j]/len(orb_pruned)*100
# Bar chart excluding invalids
rows = [ '50cm < err','25 cm < err < 50cm', '10cm < err < 25cm','err < 10cm']
data = [a,
        b,
        c,
        d]

columns = ('x', 'y', 'z')

values = np.arange(0, 1)
value_increment = 1000

# Get some pastel shades for the colors
colors = plt.cm.Reds(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x) for x in data[row]])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("%")
plt.yticks([])
plt.xticks([])
plt.title('Relative amounts of samples for each error range (excluding invalids)')
plt.show()

# Bar chart including invalids
a = [0,0,0]
b = [0,0,0]
c = [0,0,0]
d = [0,0,0]
for j in range(0,3):
    for i in range(0,len(orb_pruned)):
        if orb_pruned[i][j] < 0.1:
            a[j] += 1
        elif orb_pruned[i][j] < 0.25:
            b[j] += 1
        elif orb_pruned[i][j] < 0.5:
            c[j] += 1
        else:
            d[j] += 1
    a[j] = a[j]/(len(orb_pruned)+len(del_orb)-begin)*100
    b[j] = b[j]/(len(orb_pruned)+len(del_orb)-begin)*100
    c[j] = c[j]/(len(orb_pruned)+len(del_orb)-begin)*100
    d[j] = d[j]/(len(orb_pruned)+len(del_orb)-begin)*100
rows = [ '50cm < err','25 cm < err < 50cm', '10cm < err < 25cm','err < 10cm','invalid']
inval_rat = (len(del_orb)-begin)/(len(orb_pruned)+len(del_orb)-begin)*100
data = [[inval_rat,inval_rat,inval_rat],
        a,
        b,
        c,
        d]

columns = ('x', 'y', 'z')

values = np.arange(0, 1)
value_increment = 1000

# Get some pastel shades for the colors
colors = plt.cm.Reds(np.linspace(0, 0.5, len(rows)))
n_rows = len(data)
print(n_rows)
index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%1.1f' % (x) for x in data[row]])
# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()

# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)

plt.ylabel("%")
plt.yticks([])
plt.xticks([])
plt.title('Relative amounts of samples for each error range (including invalids)')
plt.show()

