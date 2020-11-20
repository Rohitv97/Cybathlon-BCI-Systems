"""
Convert EPFL dataset to npy arrays
"""
import numpy as np
import pyedflib as edf
from glob import glob
import mne
import matplotlib.pyplot as plt


def load(fname, tasks):
    print(fname)
    gdf = mne.io.read_raw_gdf(fname, preload=True)
    # print(gdf.annotations)
    # print(gdf.annotations.duration)
    # print(gdf.info)
    # print(gdf.info['ch_names'])
    fs = gdf.info['sfreq']
    data = gdf.get_data(return_times=False)
    print(data.shape)  # nchan x nsamp
    events, event_dict = mne.events_from_annotations(gdf)
    new_event_dict = {}
    for event in event_dict.keys():
        if event in tasks.keys():
            new_event_dict[event_dict[event]] = tasks[event][1]
            print(tasks[event][0], np.sum(events[:, -1] == event_dict[event]))
    print(event_dict)
    print(new_event_dict)
    #print(events[:, -1])
    # (1, 2, 12, 7), 9, 4, [11, 6], (1, 2, 12, 7), 10, 5, [11, 6], ...
    # (1, 32769, 'fixation', 33554), 'task', 33538, [781, 33549], (1, 32769, 'fixation', 33554), 'task', 33539, [781, 33549]
    # Ignore the huge ones
    # (1, 12), 9, [11], (1, 12), 10, [11], ...
    # {(1, 'fixation'), 'task', [781]}, {(1, 'fixation'), 'task', [781]}
    data[-1, :] = 0  # empty trigger channel
    start, ev_type = None, -1
    for ev in events:
        ev_smp = ev[0]
        if ev[-1] in new_event_dict.keys():  # start of task: this is when the subject gets told to start thinking
            start = ev_smp
            ev_type = ev[-1]
        elif ev[-1] == event_dict['33549'] and start is not None and ev_type > 0:  # Events last until a 'trial_end' cue appears (code 33549)
            data[-1, start:ev_smp] = new_event_dict[ev_type] # it's not a binary trigger.
            start = None  # reset
            ev_type = -1
    return data, fs


if __name__ == '__main__':
    root = 'D:/epfl_pilot/'
    channels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1',
                'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4']  # Not needed. Just for reference
    tasks_dict = {'769': ('leftHand', 1),
                  '770': ('rightHand', 2),
                  '771': ('feet', 3),
                  '772': ('tongue', 4),
                  '773': ('bothHands', 5),
                  '783': ('rest', 6),
                  #'786': 'fixationCross',
                  #'781': ('feedback', 9)  # trial lasts 4 seconds from here
                  #'33549': ('trial_end', 10)
                  }  # dict to convert triggers from EPFL to task names
    fs = 512.  # Also just for reference
    pilot_paths = glob('%s*VE' % root)
    print(pilot_paths)
    for pilot in pilot_paths:
        pname = pilot.split('\\')[-1]
        print(pname)
        dates = sorted(glob('%s/%s_*' % (pilot, pname)))
        print(len(dates), dates)
        for date in dates:
            files = sorted(glob('%s/*offline*gdf' % date))
            print(len(files), files)
            if len(files) == 0:
                continue
            for f, fname in enumerate(files):
                data, fs2 = load(fname, tasks_dict)
                print(fname, fs, fs2)
                assert fs2 == fs, "The sampling frequency for %s is not %.1f" % (fname, fs)
                save_fname = root + 'npys/'+ fname.split('\\')[-1][:-4]
                print('Saving file to:', save_fname)
                np.save(save_fname, data.T)
                #if f == 0:
                #    plt.plot(data[-1, :])
                #    plt.show()
                #    exit()