#!/usr/bin/env python3

from __future__ import print_function, division

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
from statannotations_new.Annotator import Annotator
#import statannotations


if __name__ == '__main__':
    df = joblib.load('model_analysis_df.pkl')
    dfm = joblib.load('model_analysis_motion_context.pkl')
    dfc = joblib.load('model_analysis_color_context.pkl')

    accs_before = df.iloc[2:4]
    accs_after = df.iloc[4:]

    accs_before = accs_before.rename({'acc_motion_before': 'motion', 'acc_color_before': 'color'})
    accs_before.index.name = 'acc_message'
    accs_before = accs_before.rename(columns=(lambda x: int(x.replace('model', ''))))
    accs_before.columns.name = 'model'

    new_index = []
    for s in accs_after.index:
        s_parts = s.split('_')
        new_s = [s_parts[1], int(s_parts[3])]
        if len(s_parts) == 5:
            new_s.append(50)
        else:
            new_s.append(100)
        new_index.append(tuple(new_s))
    accs_after = accs_after.set_index(pd.MultiIndex.from_tuples(new_index, names=['acc_message', 'node_id', 'prune_factor'])).sort_index()
    accs_after = accs_after.rename(columns=(lambda x: int(x.replace('model', ''))))
    accs_after.columns.name = 'model'

    motion_flows = dfm.iloc[0]
    color_flows = dfc.iloc[1]

    mf_ho = np.array([mf[-2] for mf in motion_flows])
    mf_hh = np.array([mf[-3] for mf in motion_flows])
    cf_ho = np.array([cf[-2] for cf in color_flows])
    cf_hh = np.array([cf[-3] for cf in color_flows])

    flow_df = pd.DataFrame(np.vstack((mf_ho, mf_hh, cf_ho, cf_hh)), columns=pd.Index(np.arange(4), name='node_id'),
                           index=pd.MultiIndex.from_tuples([(msg, flow_type, model) for msg in ['motion', 'color']
                                                                                    for flow_type in ['ho', 'hh']
                                                                                    for model in range(10)],
                                                           names=['flow_message', 'flow_type', 'model']))

    ho_flow = flow_df.xs('ho', level='flow_type')
    hh_flow = flow_df.xs('hh', level='flow_type')
    max_ho_flow = ho_flow.agg('idxmax', axis=1)
    max_hh_flow = hh_flow.agg('idxmax', axis=1)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)
    accs_dfs = []
    for i, msg in enumerate(['color', 'motion']):
        max_motion_ho_flow = max_ho_flow.xs(msg, level='flow_message')
        max_motion_ho_flow.name = 'node_id'
        accs_df = accs_after.stack().unstack(level=['acc_message', 'prune_factor']).swaplevel(axis=0)
        max_motion_accs = accs_df.loc[max_motion_ho_flow.reset_index().apply(tuple, axis=1)].stack(level=(0, 1))

        max_motion_accs = max_motion_accs.unstack(level='prune_factor').droplevel('node_id').swaplevel().sort_index()
        max_motion_accs[0] = accs_before.stack()
        max_motion_accs = max_motion_accs.sort_index(axis=1).stack()
        max_motion_accs.name = 'acc'
        max_motion_accs = max_motion_accs.reset_index()
        max_motion_accs['flow_message'] = msg

        plot_params = dict(data=max_motion_accs, x='acc_message', y='acc', hue='prune_factor')
        g = sns.boxplot(ax=axs[i], data=max_motion_accs, x='acc_message', y='acc', hue='prune_factor')

        pairs = [[('color', 0), ('color', 50)],
                 [('color', 0), ('color', 100)],
                 [('motion', 0), ('motion', 50)],
                 [('motion', 0), ('motion', 100)]]

        annotator = Annotator(g, pairs, **plot_params)
        annotator.configure(test='Mann-Whitney')
        annotator.apply_and_annotate()

    plt.show()
