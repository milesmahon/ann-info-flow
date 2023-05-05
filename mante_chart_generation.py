import numpy as np
import pandas as pd
import torch
import ast

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
            if line.startswith('y_info_flows:'):
                y_info_flows = ast.literal_eval(line[13:].replace('array(', '').replace(')', '').replace(' ', ''))
                index_max_color_flow = np.argmax(y_info_flows[9])

    return [i, accuracy/10000, acc_motion, acc_color, index_max_motion_flow, index_max_color_flow]


data = []
for i in range(50):
    data.append(get_data_before(i))
df = pd.DataFrame(np.array(data), columns=['net_number', 'accuracy', 'acc_motion', 'acc_color', 'index_max_motion', 'index_max_color'])
df.to_csv('mante_nets/before_stats.csv')

# Files we have:
# model files (.pth)
# analysis files for each base model (before prune)
# analysis files for each pruned model (after prune)
