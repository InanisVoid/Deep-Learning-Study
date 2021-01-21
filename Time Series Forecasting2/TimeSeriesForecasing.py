import os 
import IPython
import IPython.display 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import tensorflow as tf
import datetime
import seaborn as sns

mpl.rcParams['figure.figsize']=(8,6)
mpl.rcParams['axes.grid']=False

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
df = df[5::6]
date_time=pd.to_datetime(df.pop('Date Time'),format='%d.%m.%Y %H:%M:%S')
# print(df)

# plot_cols=['T (degC)','p (mbar)','rho (g/m**3)']
# plot_features=df[plot_cols]
# plot_features.index=date_time
# plot_features.plot(subplots=True)
# plot_features=df[plot_cols][:480]
# plot_features.index=date_time[:480]
# plot_features.plot(subplots=True)
# plt.show()
wv = df['wv (m/s)']
bad_wv= wv==-9999.0
wv[bad_wv]=0.0
max_wv=df['max. wv (m/s)']
bad_max_wv= max_wv==-9999.0
max_wv[bad_max_wv]=0.0


# plt.hist2d(df['wd (deg)'], df['wv (m/s)'], bins=(50, 50), vmax=400)
# plt.colorbar()
# plt.xlabel('Wind Direction [deg]')
# plt.ylabel('Wind Velocity [m/s]')
# plt.show()

wv = df.pop('wv (m/s)')
max_wv = df.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = df.pop('wd (deg)')*np.pi / 180

# Calculate the wind x and y components.
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# Calculate the max wind x and y components.
df['max Wx'] = max_wv*np.cos(wd_rad)
df['max Wy'] = max_wv*np.sin(wd_rad)

# plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
# plt.colorbar()
# plt.xlabel('Wind X [m/s]')
# plt.ylabel('Wind Y [m/s]')
# ax = plt.gca()
# ax.axis('tight')
# # plt.show()

timestamp_s = date_time.map(datetime.datetime.timestamp)
# print(timestamp_s)

day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))


# print(df.describe())
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

# print(df.shape)
num_features = df.shape[1] #19 columns

# print(column_indices)
# print(num_features)

train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


# df_std = (df - train_mean) / train_std
# df_std = df_std.melt(var_name='Column', value_name='Normalized')
# plt.figure(figsize=(12, 6))
# ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
# _ = ax.set_xticklabels(df.keys(), rotation=90)


# plt.show()
# # prin

