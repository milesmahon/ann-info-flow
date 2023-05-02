# multiprocess data generation for mante data
from analyze_info_flow import analyze_info_flow_rnn
from nn_rnn_mante import train_new_rnn, RNN
from multiprocessing import Pool
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 4
num_layers = 1
batch_size = 10000
input_size = 3  # (motion (float), color (float), context (bool/int))
output_size = 2  # (-1, 1) one-hot encode

def gen_network(i):
    FILE = f"mante_nets/mante_rnn_{i}.pth"
    print(f'start gen{i}')
    model, _, _, _, _, _, _ = train_new_rnn()
    torch.save(model.state_dict(), FILE)

def analyze_network(i):
    print(f'start analysis{i}')
    FILE = f"mante_nets/mante_rnn_{i}.pth"
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size).to(device)
    model.load_state_dict(torch.load(FILE, map_location=device))
    z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy, acc_motion, acc_color, acc_context = \
        analyze_info_flow_rnn(model, 'linear-svm', model_name=f'model_{i}')
    with open(f'mante_nets/mante_rnn_analysis_{i}.txt', 'w') as f:
        f.write(f'z_mis: {z_mis}\n')
        f.write(f'z_info_flows: {z_info_flows}\n')
        f.write(f'z_info_flows_weighted: {z_info_flows_weighted}\n')
        f.write(f'y_mis: {y_mis}\n')
        f.write(f'y_info_flows: {y_info_flows}\n')
        f.write(f'y_info_flows_weighted: {y_info_flows_weighted}\n')
        f.write(f'accuracy: {accuracy}\n')
        f.write(f'acc_motion: {acc_motion}\n')
        f.write(f'acc_color: {acc_color}\n')
        f.write(f'acc_context: {acc_context}\n')


# NEXT:
# for each network,
#   prune each of the 4 self edges and save as mante_rnn_{i}_pruned_{j}.pth with j in range(4)
#   test the pruned networks
#   get pre-prune accuracy for each task from analysis file
#   get post-prune accuracy for each task
#   get info flow of each task on each pruned edge from original analysis file
#   save this as a dataframe
#   use praveen's seaborn code to plot the dataframe


if __name__ == '__main__':
    with Pool(8) as p:
        p.map(gen_network, range(50))

    with Pool(8) as p:
        p.map(analyze_network, range(50))