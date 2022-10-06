import pandas as pd
from pandas import DataFrame
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def give_state_name(peak_cur):
    if len(peak_cur) == 2:
        state_name = ['L','H']
    elif len(peak_cur) ==3:
        state_name = ['L','M','H']
    else:
        state_name = ['L']
        for i in range(1,len(peak_cur)-1):
            state_name.append('M'+str(i))
        state_name.append('H')
    return state_name

def trim_ht(bin_df):
    # ------去头去尾-------
    first_row_index = bin_df.index.values[0]
    last_row_index = bin_df.index.values[len(bin_df) - 1]

    if len(bin_df) == 1:
        bin_df.drop(labels=first_row_index, axis=0, inplace=True)
    else:
        bin_df.drop(labels=[first_row_index, last_row_index], axis=0, inplace=True)

    return bin_df


def trim_h(bin_df):
    # ------去头-------
    first_row_index = bin_df.index.values[0]
    bin_df.drop(labels=first_row_index, axis=0, inplace=True)

    return bin_df


def trim_t(bin_df):
    # ------去尾-------
    last_row_index = bin_df.index.values[len(bin_df) - 1]
    bin_df.drop(labels=last_row_index, axis=0, inplace=True)

    return bin_df


def split_states(bin_df, title, peak_cur, state_name):
    df_states = []
    print('---------------------' + title + '-----------------------------')
    for i in range(len(peak_cur)):
        tmpdf = bin_df[bin_df['status'] == i]['t'].sort_values().to_frame()
        tmpdf['No.'] = range(1, len(tmpdf) + 1)
        tmp_sum = tmpdf['t'].sum()
        tmp_count = tmpdf['t'].count()
        print('State(', state_name[i], ') sum & count:', round(tmp_sum,4), '\t', tmp_count)
        df_states.append(tmpdf)

    return df_states


def kneecaps(dfstates):
    class Knees:
        def __init__(self, df, best_knee):
            self.df = df
            self.best = best_knee
            self.manual = [0, 0]

    knees_cls_lst = []
    for i in dfstates:
        i.reset_index(drop=True, inplace=True)
        i_knees = KneeLocator(i['No.'], i['t'], curve='convex', direction='increasing', online=True)
        i_best_knee = [i_knees.knee, i_knees.knee_y]

        i_knees_df = DataFrame(columns=['x', 'y'])
        tmp = list(i_knees.all_knees)
        tmp.sort()
        i_knees_df['x'] = tmp
        tmp = i_knees.all_knees_y
        tmp.sort()
        i_knees_df['y'] = tmp
        knees_cls_lst.append(Knees(i_knees_df, i_best_knee))

    return knees_cls_lst


def flatten(old_bin_df, knee_point, knee_tag, fstyle):
    resolution = fstyle.resolution
    new_bin_df = DataFrame(columns=['t', 'status', 't_start', 't_end','tag'])
    lastindex = old_bin_df.index[0]
    for i in old_bin_df.index:
        tmp = old_bin_df.loc[i].copy()
        if tmp['status'] == knee_tag:
            if tmp['t'] > knee_point[1]:
                new_bin_df = new_bin_df.append(tmp)
            else:
                tmp['status'] = old_bin_df.loc[lastindex, 'status']
                new_bin_df = new_bin_df.append(tmp)
        else:
            new_bin_df = new_bin_df.append(tmp)
        lastindex = i

    # print(new_bin_df)

    tag = new_bin_df.index[0]
    last_i = tag
    last_i_t = new_bin_df.loc[last_i, 't']
    last_i_start = new_bin_df.loc[last_i, 't_start']
    for i in new_bin_df.index:
        if i == tag:
            pass
        else:
            i_start = new_bin_df.loc[i, 't_start']
            if new_bin_df.loc[tag, 'status'] == new_bin_df.loc[i, 'status'] and i_start - last_i_start == last_i_t:
                tmp = new_bin_df.loc[tag, 't'] + new_bin_df.loc[i, 't']
                last_i = i
                last_i_t = new_bin_df.loc[i, 't']
                last_i_start = new_bin_df.loc[i, 't_start']
                new_bin_df.loc[tag, 't'] = tmp
                new_bin_df.loc[tag, 't_end'] = new_bin_df.loc[i, 't_end']
                new_bin_df.drop(labels=i, axis=0, inplace=True)
            elif i_start - last_i_start != last_i_t:
                new_bin_df.drop(labels=tag, axis=0, inplace=True)
                tag = i
                last_i = i
                last_i_t = new_bin_df.loc[last_i, 't']
                last_i_start = new_bin_df.loc[last_i, 't_start']
            else:
                new_bin_df.loc[tag, 'tag'] ='{0}-{1}'.format(new_bin_df.loc[tag,'status'],new_bin_df.loc[i,'status'])
                tag = i
                last_i = i
                last_i_t = new_bin_df.loc[last_i, 't']
                last_i_start = new_bin_df.loc[last_i, 't_start']
                continue

    if 'old_index' not in new_bin_df:
        new_bin_df['old_index'] = new_bin_df.index

    # new_bin_df['t_end'] = new_bin_df['t'].cumsum().apply(lambda x: x+t_start)
    # new_bin_df['t_start'] = new_bin_df['t_end'] - new_bin_df['t']
    # new_bin_df['t_end'] = new_bin_df['t_end'].apply(lambda x: x-0.0001)

    # print(new_bin_df)
    randflat = np.random.rand(1)
    new_bin_df.to_csv('flatten'+str(randflat)+'.csv')
    old_bin_df.to_csv('before flatten' + str(randflat) + '.csv')

    return new_bin_df


def plot_t_count_knees(dfstates, knees, peak_cur, statename, filename, title):
    n = len(dfstates)
    # 两个subplot
    for i in range(n):
        plt.figure(figsize=(11, 4.8))
        plt.subplot(121)
        plt.plot(dfstates[i]['No.'], dfstates[i]['t'])
        plt.scatter(x=knees[i].df['x'], y=knees[i].df['y'], c='b', s=50, marker='^', alpha=1)
        plt.scatter(x=knees[i].best[0], y=knees[i].best[1], c='r', s=30, marker='o', alpha=1)
        plt.subplot(122)
        plt.plot(dfstates[i]['No.'], dfstates[i]['t'])
        plt.scatter(x=knees[i].df['x'], y=knees[i].df['y'], c='b', s=50, marker='^', alpha=1)
        plt.scatter(x=knees[i].best[0], y=knees[i].best[1], c='r', s=30, marker='o', alpha=1)
        plt.xscale('log')
        plt.yscale('log')
        plt.title(title + ' State(' + statename[i] + ')', loc='right')
        plt.savefig('I-count-'+statename[i]+'-' + filename+'-' + title + '.png')
        print('Please select a Knee on State(' + statename[i] + ')')
        man = plt.ginput(1, timeout=1800)
        man_index = dfstates[i]['No.'][dfstates[i]['No.'].values == np.floor(man[0][0])].index.values[0]
        knees[i].manual = [dfstates[i]['No.'].loc[man_index], dfstates[i]['t'].loc[man_index]]
        print('The knee on ' + title + ' State(' + statename[i] + ') chosen is:', knees[i].manual)

    # plt.show()

    return


def choose_knee(knees, statename):
    knee_chosen = []
    print('----------------------\nStates you need to choose knee on:')
    for i in range(len(statename)):
        print(str(i), '.', statename[i])
    state_tag = int(input('Please enter:'))
    knee_tag = int(input('Knee option (0.BEST Knee; n.No.Knee(L to R); -1.Manual):'))
    if knee_tag == 0:
        knee_chosen = knees[state_tag].best
    elif knee_tag == -1:
        knee_chosen = knees[state_tag].manual
    else:
        knee_chosen = knees[state_tag].df.iloc[knee_tag - 1]
        knee_chosen = knee_chosen.tolist()

    return knee_chosen, state_tag


def plot_flatten(combined_data, flatten_df, peak_cur,filename, title):
    fig_data = DataFrame(columns=['t', 'I'])

    for i in flatten_df.index:
        t_start = flatten_df.loc[i, 't_start']
        t_end = flatten_df.loc[i, 't_end']

        fig_data = fig_data.append({'t': t_start, 'I': peak_cur[flatten_df.loc[i, 'status']]}, ignore_index=True)
        fig_data = fig_data.append({'t': t_end, 'I': peak_cur[flatten_df.loc[i, 'status']]}, ignore_index=True)

    ax = combined_data.plot(x="t", y='I', kind='line')
    fig_data.plot(x='t', y='I', legend=False, ax=ax)
    plt.title(title, loc='right')
    plt.savefig('I-t-' + filename + '-' + title + '.png')

    return


def split(start_t, original_df, flatten_df):
    split_df = DataFrame(columns=['t', 'status', 'old_index', 't_start', 't_end','tag'])

    tag = original_df.index[0]

    for i in original_df.index:
        if i == tag:
            continue
        else:
            pass

        if i not in flatten_df.index:
            if tag in flatten_df.index:  # 补头
                tmp = original_df.loc[tag].copy()
                split_df = split_df.append(tmp)
                # print('split补头:',tag,i)

            tmp = original_df.loc[i].copy()
            split_df = split_df.append(tmp)
        else:
            if tag not in flatten_df.index:  # 去尾
                split_df.drop(labels=tag, inplace=True)
                # print('split去尾:',tag,i)

        tag = i

    split_df['old_index'] = split_df.index
    # split_df.to_csv('split_df.csv')

    return split_df


def plot_t_hist(bin_df, filename, title):
    df = bin_df.copy()
    for i in df.index:
        if df.loc[i, 't'] < 0.00055:
            # print('yes', df.loc[i, 't'])
            df.drop(labels=i, inplace=True)
    if len(df) == 0:
        print('Hist diagram error: All signals in', filename + '-'+title, 'are noise and have been deleted.')
        return
    else:
        # print(len(df),title)
        df_bin = int(1 + 3.322 * np.log10(len(df)))
        df.hist(column='t', bins=df_bin)
        plt.title(title,loc='right')

        hist_data = pd.cut(df['t'], df_bin).value_counts().sort_index()
        hist_df = pd.DataFrame(columns=['t', 'count'])
        hist_df['t'] = hist_data.index
        hist_df['t'] = hist_df['t'].apply(lambda x: x.mid)
        hist_df['count'] = hist_data.values

        plt.savefig('t-hist-' + filename +'-'+ title + '.png')
        hist_df.to_csv('t-hist-' + filename + '-'+title + '.csv', index=False)

    return hist_df


def exp_stat(x, a, r):
    return a * np.exp(0 - r * x)


def split_states_adv(bin_df, peak_cur,statename,filename,title):
    bin_df = trim_ht(bin_df)
    lens = len(peak_cur)
    res = []
    for i in range(lens):
        for j in range(lens):
            if j == i:
                continue
            else:
                tmpdf = bin_df[bin_df['tag'] == '{0}-{1}'.format(str(i), str(j))]['t'].sort_values().to_frame()
                tmpdf['No.'] = range(1, len(tmpdf) + 1)
                tmp_sum = tmpdf['t'].sum()
                tmp_count = tmpdf['t'].count()
                print('State(', statename[i], ' to ',statename[j],') sum & count:', round(tmp_sum,5), '\t', tmp_count)
                tmpdf.to_csv('t-count-' + statename[i]+ ' to ' + statename[j] + '-'+filename+'-'+title+'.csv',
                                      index=False)
                plot_t_hist(tmpdf,statename[i]+'-'+statename[j]+'-'+filename,title+'-States('+statename[i]+')')

                res.append(tmpdf)
    plt.show()

    return res
