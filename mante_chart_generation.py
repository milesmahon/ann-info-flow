import numpy as np
import pandas as pd
import torch
import ast

from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Z is motion
# Y is color
# U is true
# import warnings
# warnings.simplefilter("ignore", DeprecationWarning)

# fields = ['net_number', 'accuracy', 'acc_motion', 'acc_color',
#           'pruned_edge', 'info_flow_motion', 'info_flow_color', 'acc_motion_after', 'acc_color_after']

def get_data_before(i):
    with open(f"mante_nets/mante_rnn_analysis_{i}.txt", "r") as f:
        for line in f:
            if line.startswith('accuracy'):
                accuracy = float(line.split(' ')[1])
            if line.startswith('acc_motion'):
                acc_motion = float(line.split(' ')[1])
            if line.startswith('acc_color'):
                acc_color = float(line.split(' ')[1])
            if line.startswith('z_info_flows:'):
                z_info_flows = ast.literal_eval(line[13:].replace('array(', '').replace(')', '').replace(' ', ''))
                index_max_motion_flow = np.argmax(z_info_flows[9])  # 9 is last layer before output
                motion_info_flows = z_info_flows[9]
            if line.startswith('y_info_flows:'):
                y_info_flows = ast.literal_eval(line[13:].replace('array(', '').replace(')', '').replace(' ', ''))
                index_max_color_flow = np.argmax(y_info_flows[9])
                color_info_flows = y_info_flows[9]

    return [i, accuracy/10000, acc_motion, acc_color, index_max_motion_flow, index_max_color_flow, color_info_flows, motion_info_flows]


data = []
motion_flows = []
color_flows = []
for i in range(50):
    model_stats = get_data_before(i)
    data.append(model_stats)
    for flow in model_stats[7]:
        motion_flows.append(flow)
    for flow in model_stats[6]:
        color_flows.append(flow)

df = pd.DataFrame(np.array(data), columns=['net_number', 'accuracy', 'acc_motion', 'acc_color', 'index_max_motion', 'index_max_color', 'color_info_flows', 'motion_info_flows'])
df.to_csv('mante_nets/specialization_stats.csv')

# # Z is motion
# # Y is color
cmap = plt.colormaps['RdBu']
norm = plt.Normalize(0, 1.8)

ratios = [np.arctan(norm(x)/norm(y)) for x in motion_flows for y in color_flows]
plt.hist(ratios, bins=100, alpha=0.5)

plt.xlabel("Ratio of motion to color flow by edge", fontsize=20)

ax = plt.gca()
# ax.plot([0,1], [0,1], '--k', transform=ax.transAxes)
ax.set_title("Motion and color flow by edge", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=14)
plt.savefig(fname='mante_nets/specialization.png', dpi=300)

# Files we have:
# model files (.pth)
# analysis files for each base model (before prune)
# analysis files for each pruned model (after prune)
