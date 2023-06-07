#!/usr/bin/python3

import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import ffio

video_fns = [fn for fn in os.listdir('data_raw') if 'mp4' in fn]
video_time_names = ['.'.join(fn.split('.')[:-1]) for fn in video_fns]
video_times = [float(t) for t in video_time_names]

bag_fns = sorted(os.listdir('data_raw'))
bag_fns = [fn for fn in bag_fns if 'bag' in fn]
bag_basenames = [fn.split('.')[0] for fn in bag_fns]
# subset_2021-08-04-19-42-22_0_csv_raw_gimbal.csv
split_starts = [ix for (ix, fn) in enumerate(bag_fns) if int(fn.split('_')[2].split('.')[0]) == 0] + [0]

bag_collections = [bag_fns[split_starts[ix]:split_starts[ix+1]] for ix in range(len(split_starts)-1)][:-1]
bagname_collections = [bag_basenames[split_starts[ix]:split_starts[ix+1]] for ix in range(len(split_starts)-1)][:-1]

all_times = []
for run in bag_collections:
    bag_times = []
    for fn in run:
        time_str = fn.split('_')[1]
        time_tuple = time.strptime(time_str, '%Y-%m-%d-%H-%M-%S')
        time_epoch = time.mktime(time_tuple)
        bag_times.append(time_epoch)
    all_times.append(bag_times)

base_time = all_times[0][0]
for (ix, run) in enumerate(all_times):
    plt.scatter([r - base_time for r in run], [ix] * len(run))

plt.scatter([v + 14400 - base_time for v in video_times], [30] * len(video_times))
plt.vlines([v + 14400 - base_time for v in video_times], 0, 60)
plt.show()

video_times = np.array([v + 14400 - base_time for v in video_times])
run_start_times = np.array([r[0] - base_time for r in all_times])

bag_ix_matches = []

for ix in range(len(run_start_times)):
    min_delta = np.min(np.abs(run_start_times[ix] - video_times))
    if min_delta < 5:
        bag_ix_matches.append(np.argmin(np.abs(run_start_times[ix] - video_times)))
    else:
        bag_ix_matches.append(None)

good_count = 0
for (ix, match) in enumerate(bag_ix_matches):
    if match is None:
        continue
    good_count += 1
    plt.scatter(run_start_times[ix], [ix])
    plt.vlines([video_times[match]], 0, 60)
    

print(good_count)
plt.show()


for (run_index, video_index) in enumerate(bag_ix_matches):
    print('Processing run %d of %d' % (run_index + 1, len(bag_ix_matches)))
    if video_index is None:
        continue

    vid_fn = video_time_names[video_index]
    frame_fn = 'frame_times_%s.txt' % vid_fn
    df_frames = pd.read_csv('data_raw/%s' % frame_fn, names=['%time'])

    #frame_basetime = df_frames.time.iloc[0]
    #rel_times = df_frames.time - frame_basetime
    #nominal_times = np.arange(len(rel_times))/30
    #frame_offsets = rel_times - nominal_times


    df_frames['%time'] = pd.to_datetime(df_frames['%time'], unit='s')
    df_frames = df_frames.set_index('%time')
    base_time = df_frames.index[0]
    df_frames.index = df_frames.index - base_time
    df_frames['frame_index'] = np.arange(len(df_frames))

    df = pd.DataFrame()
    for bix in range(len(bag_collections[run_index])):
        bag_fn = bagname_collections[run_index][bix]
        try:
            df_rc = pd.read_csv('data_csv/%s_csv_raw_rc.csv' % bag_fn)
            df_position = pd.read_csv('data_csv/%s_csv_raw_localposition.csv' % bag_fn)
            df_velocity = pd.read_csv('data_csv/%s_csv_raw_velocity.csv' % bag_fn)
            df_angular_velocity = pd.read_csv('data_csv/%s_csv_raw_angularvelocity.csv' % bag_fn)
            df_attitude = pd.read_csv('data_csv/%s_csv_raw_attitude.csv' % bag_fn)
        except Exception as ex:
            print(ex)
            # sometimes there's a problem with the bag extraction and we get an empty csv?
            continue

        df_rc['%time'] = pd.to_datetime(df_rc['%time'], unit='ns') - base_time
        df_rc = df_rc.drop(columns=['field.header.seq', 'field.header.stamp', 'field.header.frame_id'])
        df_rc = df_rc.rename(columns={'field.axes0': 'axes0', 'field.axes1': 'axes1', 'field.axes2': 'axes2', 'field.axes3': 'axes3', 'field.axes4': 'axes4', 'field.axes5': 'axes5'})

        df_position['%time'] = pd.to_datetime(df_position['%time'], unit='ns') - base_time
        df_position = df_position.drop(columns=[ 'field.header.seq', 'field.header.stamp', 'field.header.frame_id'])
        df_position = df_position.rename(columns={'field.point.x': 'local_x', 'field.point.y': 'local_y','field.point.z': 'local_z'})

        df_velocity['%time'] = pd.to_datetime(df_velocity['%time'], unit='ns') - base_time
        df_velocity = df_velocity.drop(columns=['field.header.seq', 'field.header.stamp', 'field.header.frame_id'])
        df_velocity = df_velocity.rename(columns={'field.vector.x': 'vx', 'field.vector.y': 'vy', 'field.vector.z': 'vz'})

        df_angular_velocity['%time'] = pd.to_datetime(df_angular_velocity['%time'], unit='ns') - base_time
        df_angular_velocity = df_angular_velocity.drop(columns=['field.header.seq', 'field.header.stamp', 'field.header.frame_id'])
        df_angular_velocity = df_angular_velocity.rename(columns={'field.vector.x': 'omega_x', 'field.vector.y': 'omega_y', 'field.vector.z': 'omega_z'})

        df_attitude['%time'] = pd.to_datetime(df_attitude['%time'], unit='ns') - base_time
        df_attitude = df_attitude.drop(columns=['field.header.seq', 'field.header.stamp', 'field.header.frame_id'])
        df_attitude = df_attitude.rename(columns={'field.vector.x': 'omega_x', 'field.vector.y': 'omega_y', 'field.vector.z': 'omega_z'})

        df_rc = df_rc.set_index('%time')
        df_position = df_position.set_index('%time')
        df_velocity = df_velocity.set_index('%time')
        df_angular_velocity = df_angular_velocity.set_index('%time')
        df_attitude = df_attitude.set_index('%time')

        full_df = df_rc.join(df_position, how='outer').join(df_velocity, how='outer').join(df_angular_velocity, how='outer').join(df_attitude, how='outer')
        df = pd.concat([df, full_df])

    df_combined = df_frames.join(df, how='outer').interpolate(method='nearest')
    df_30hz = df_combined.resample('33333us').nearest()


    bag_start_ix = np.max(df_30hz.notna().idxmax()) + pd.Timedelta(seconds=1)
    df_30hz = df_30hz[bag_start_ix:]
    df_30hz.index = np.array(df_30hz.index.astype(np.int64)) / 1e9

    bag_end_ix = np.min(df_30hz.notna().idxmin()[df_30hz.isnull().any()])
    df_30hz = df_30hz[:bag_end_ix]

    if not os.path.exists('frames/%s' % vid_fn):
        os.mkdir('frames/%s' % vid_fn)
    df_30hz.to_csv('frames/%s/training_inputs_w_attitude.csv' % vid_fn)


    video_reader = ffio.FFReader('data_raw/%s.mp4' % vid_fn)
    row_index = 0
    for row_index in range(len(df_30hz)):
        if row_index % 500 == 0: print('Frame %d of %d' % (row_index, len(df_30hz)))
        frame_index = df_30hz.frame_index.iloc[row_index]

        while video_reader.frame_num < frame_index:
            (ret, frame) = video_reader.read()
            if not ret:
                break
        im = Image.fromarray(frame)
        resize_ratio = 0.2
        #im_smaller = im.resize( [int(resize_ratio*s) for s in im.size], resample=Image.BILINEAR)
        im_smaller = im.resize( [int(resize_ratio*s) for s in im.size])
        im_smaller.save('frames/%s/%06d.png' % (vid_fn, row_index))
        row_index += 1





