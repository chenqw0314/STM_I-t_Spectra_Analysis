import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from it_analysis import split_states, kneecaps, flatten, plot_t_count_knees, choose_knee, plot_flatten, \
    trim_ht, plot_t_hist, split, split_states_adv, give_state_name
from data_proc import load_data, plt_I_t_hist, drift_corr, save2csv, raw_to_bin, delete

# --------------全局变量---------------
class Fstyle:
    def __init__(self, resolution, header_size, noise):
        self.resolution = resolution
        self.header_size = header_size
        self.noise = noise

nanonis_style = Fstyle(resolution=0.0005, header_size=13, noise=2)

# --------------数据目录---------------
rootdir = os.getcwd()
rootfiles = os.listdir(rootdir)
datadir = os.getcwd() + '\\data'
datafiles = os.listdir(datadir)

# --------------载入数据---------------
prev_datas = []
for f in rootfiles:
    try:
        if re.search(r'(.*)_(.*).csv',f).group(1) ==  'datapd':
            prev_datas.append(f)
    except:
        pass

len_prevdata = len(prev_datas)
while True:
    if len_prevdata > 0:
        print('Find previous data.\n')
        for i in range(len_prevdata):
            print(str(i+1)+'.'+prev_datas[i])
        data_num = int(input('0.Exit\nPlease choose which one to load:'))
        if data_num != 0:
            combined_data = pd.read_csv(rootdir+'\\'+prev_datas[data_num-1])
            filename = re.search(r'(.*)_(.*).csv',prev_datas[data_num-1]).group(2)
            histdata, peak_I, setpoints = plt_I_t_hist(combined_data, filename,nanonis_style) # 作I-t曲线和电流直方图
            len_prevdata = 0
            break
        else:
            len_prevdata = 0
    else:
        combined_data, filename = load_data(datadir, datafiles, nanonis_style)
        histdata, peak_I, setpoints = plt_I_t_hist(combined_data, filename,nanonis_style) # 作I-t曲线和电流直方图
        while True:
            option = int(input('Choose (1.Drift correction;2.Delete;0.Continue):'))
            if option == 1:
                combined_data = drift_corr(combined_data, filename, nanonis_style) # 漂移校正
            elif option ==2:
                delete_time = [float(n) for n in input('Delete start & end time:').split(',')]
                combined_data = delete(combined_data,delete_time[0],delete_time[1],nanonis_style)
                plt_I_t_hist(combined_data, filename, nanonis_style)
            elif option ==0:
                break
            else:
                print('Wrong input.')
        save2csv(combined_data, 'datapd_', filename) # 保存文件
        break

# ---------------setpoint和态电流选择----------------------------
sp_cur = input('-------------------\nPlease enter setpoint(pA)(enter 0 to use recommended):')
# sp_cur = '0'
if sp_cur == '0':
    sp_cur = [round(n,1) for n in setpoints]
elif ',' in sp_cur:
    sp_cur = [float(n) for n in sp_cur.split(',')]
else:
    sp_cur = [float(sp_cur)]

peak_cur = input('-------------------\nPlease enter peak current(pA)(enter 0 to use recommended):')
# peak_cur = '0'
if peak_cur == '0':
    peak_cur = [round(n,1) for n in peak_I]
elif ',' in peak_cur:
    peak_cur = [float(n) for n in peak_cur.split(',')]
else:
    peak_cur = [float(peak_cur)]

statename = give_state_name(peak_cur)

# print(sp_cur,peak_cur)
# ---------------转化为态与时长文件----------------------------
bin_df = raw_to_bin(combined_data, sp_cur)
save2csv(bin_df,'bin_I_t_',filename)
plot_flatten(combined_data, bin_df,peak_cur, filename, 'RawDF')

plt.show()

start_t = bin_df.iloc[0]['t']
# tm_ht_bin_df = trim_ht(bin_df)

current_dfs = ['']
current_dfs[0] = bin_df

current_dfs_titles = ['']
current_dfs_titles[0] = 'RawDF'

i = 0
while True:
    len_j = len(current_dfs)
    dfstates_lst = list(range(len_j))
    knees = list(range(len_j))

    opt_df = ''
    for j in range(len_j):
        plot_title = current_dfs_titles[j]
        dfstates_lst[j] = split_states(current_dfs[j], plot_title, peak_cur, statename)
        try:
            knees[j] = kneecaps(dfstates_lst[j])
            plot_t_count_knees(dfstates_lst[j], knees[j],peak_cur, statename, filename, plot_title)

        except:
            print('Error.')
        finally:
            if i == 0:
                pass
            else:
                plot_flatten(combined_data, current_dfs[j], peak_cur,filename, plot_title)
            # print(dfstates_lst[j])
            for i in range(len(dfstates_lst[j])):
                plot_t_hist(dfstates_lst[j][i], filename, plot_title + '-States('+statename[i]+')')
            tmp = str(j + 1) + '.' + plot_title + '\n'
            opt_df += tmp

    print('---------------------------------------------------------------------\ncurrent Bin_dfs you have:\n', opt_df)
    plt.show()

    breaker = int(input("Do you want to continue flattening? (1. YES; 0. NO):"))

    if breaker == 0:
        print('job done.')
        for j in range(len_j):
            plot_title = current_dfs_titles[j]
            dfstates_lst[j] = split_states(current_dfs[j], plot_title, peak_cur,statename)
            for i in range(len(dfstates_lst[j])):
                dfstates_lst[j][i].to_csv('t-count-'+statename[i]+'-' + filename + '-' + plot_title + '.csv', index=False)
        print('---------------------------------------------------------------------\ncurrent Bin_dfs you have:\n',
              opt_df)
        final_df_num = int(input('Which one to do final analysis?'))
        final_df = current_dfs[final_df_num-1]
        res =split_states_adv(final_df,peak_cur,statename,filename,current_dfs_titles[final_df_num-1])

        break
    else:
        flatten_marker = int(input('Which bin_df to flatten:')) - 1
        knee_point, state_tag = choose_knee(knees[flatten_marker],statename)

        flatten_df = flatten(current_dfs[flatten_marker], knee_point, state_tag, nanonis_style)
        # print('flatten is\n',flatten_df)
        split_df = split(start_t, current_dfs[flatten_marker], flatten_df)

        last_title = current_dfs_titles[flatten_marker]
        current_dfs_titles.pop(flatten_marker)
        # print(current_dfs_titles,type(current_dfs_titles))
        current_dfs_titles.append(last_title + '+flatten')
        current_dfs_titles.append(last_title + '+split')

        current_dfs.pop(flatten_marker)
        current_dfs.append(flatten_df)
        current_dfs.append(split_df)

        i += 1
