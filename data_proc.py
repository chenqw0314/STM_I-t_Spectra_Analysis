from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


def load_data(datadir, datafiles, fstyle):
    resolution = float(fstyle.resolution)
    print('Loading data from Data directory.')
    # ----------------文件合并-------------------
    combined_data = DataFrame(columns=['t', 'I'])

    filename = datafiles[0][10:13]
    filepath = datadir + '\\' + datafiles[0]
    tmp_pd = pd.read_table(filepath, header=fstyle.header_size)

    size_all = 0
    num_of_files = len(datafiles)
    for i in range(num_of_files):
        print('\r', str(i + 1) + '/' + str(num_of_files) + ' raw data files loaded.', end='', flush=True)
        filepath = datadir + '\\' + datafiles[i]
        tmp_pd = pd.read_table(filepath, header=fstyle.header_size)
        size = len(tmp_pd)
        tmp_pd.rename(columns={'Current (A)': 'I'}, inplace=True)
        # print(tmp_pd)
        # tmp_pd['t'] = np.arange(0, resolution * size, resolution)
        # tmp_pd['t'] = tmp_pd['t'].apply(lambda x: x + resolution * size_all)
        # print(tmp_pd['t'])
        size_all += size
        combined_data = pd.concat([combined_data, tmp_pd])
    print('\n')

    filename += '-' + datafiles[num_of_files - 1][10:13] + '(' + str(num_of_files) + ')'

    combined_data['I'] = np.abs(combined_data['I'] * 1E12)  # 化成pA
    combined_data.index = range(len(combined_data))
    combined_data['t'] = np.arange(0, resolution * len(combined_data), resolution)
    # print(combined_data)

    return combined_data, filename


def plt_I_t_hist(combined_data, filename, fstyle):
    fig1 = plt.figure(num=1, figsize=(11, 4.8))
    f1ax1 = fig1.add_subplot(1, 2, 1)
    combined_data.plot(x="t", y='I', kind='line', ax=f1ax1)

    f1ax2 = fig1.add_subplot(1, 2, 2)
    cur_bins = np.arange(combined_data['I'].min(), combined_data['I'].max(), 0.1)
    histdata = f1ax2.hist(x=combined_data['I'], bins=cur_bins, log=False)
    width = fstyle.noise / 0.169
    peak_index = find_peaks(histdata[0], width=width)
    peak_I = []
    peak_count = []
    for i in peak_index[0]:
        # print(i)
        peak_I.append(histdata[1][i])
        peak_count.append(histdata[0][i])
    f1ax2.scatter(x=peak_I, y=peak_count, color='red')
    print('Find peaks at current:', peak_I)
    setpoints = []
    for i in range(len(peak_I) - 1):
        setpoints.append((peak_I[i] + peak_I[i + 1]) / 2)
    print('Recommended setpoint:', setpoints)

    plt.savefig('fig_I-t_hist_' + filename + '.png')
    plt.show()

    return histdata, peak_I, setpoints


def drift_corr(combined_data, filename, fstyle):
    input_start_end = input('\nDrift Correction(enter 0 to exit). Start I, End I:')
    if input_start_end == '0':
        print('Nothing corrected.')
        return combined_data
    else:
        start_end = [float(n) for n in input_start_end.split(',')]

        data_size = len(combined_data)
        corr = np.log(start_end[1] / start_end[0]) / (data_size * fstyle.resolution)
        combined_data['I'] = combined_data['I'] / np.exp((corr * combined_data['t']) ** 1)

        # print(combined_data)
        # combined_data.plot(x="t", y='I', kind='line')
        # cur_bins = np.arange(combined_data['I'].min(), combined_data['I'].max(), 0.1)
        # combined_data.hist(column='I', bins=cur_bins)
        plt_I_t_hist(combined_data, filename, fstyle)
        plt.show()

        return combined_data


def save2csv(pd, prefix, filename):
    pd.to_csv(prefix + filename + '.csv', index=False)


def find_interval(n, sp_cur):
    len_sp_cur = len(sp_cur)
    if n < sp_cur[0]:
        return 0
    for i in range(len_sp_cur - 1):
        if sp_cur[i + 1] > n >= sp_cur[i]:
            return i + 1
    if n > sp_cur[len_sp_cur - 1]:
        return len_sp_cur


def raw_to_bin(combined_data, sp_cur):
    time_tag = 0
    bin_df = DataFrame(columns=['t', 'status'])

    for j in range(len(combined_data)):
        present_cur = combined_data['I'][j]
        if j < len(combined_data) - 1:
            next_cur = combined_data['I'][j + 1]
        else:
            next_cur = combined_data['I'][j]

        present_tag = find_interval(present_cur, sp_cur)
        next_tag = find_interval(next_cur, sp_cur)

        if present_tag != next_tag or j == len(combined_data) - 1:
            tmp_t = combined_data['t'][j] - time_tag

            bin_df = bin_df.append([{'t': tmp_t, 'status': present_tag, 't_start': time_tag, \
                                     't_end': combined_data['t'][j],'tag': '{0}-{1}'.format(
                str(present_tag), str(next_tag))}])
            time_tag = combined_data['t'][j]

    bin_df.reset_index(drop=True, inplace=True)

    return bin_df

def delete(data, del_start, del_end, fstyle):
    resolution = fstyle.resolution
    new_data = data.copy()
    start = int(del_start/resolution)
    end = int(del_end/resolution)
    if end > len(data):
        end = len(data)
    # print(new_data)
    new_data.drop(new_data.index[start:end],axis = 0, inplace=True)
    new_data['t'] = np.arange(0, len(new_data), 1) * resolution

    return new_data