# multiprocess pruning and testing
from datasets.MotionColorDataset import MotionColorDataset
from nn_rnn_mante import RNN
from multiprocessing import Pool
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 4
num_layers = 1
batch_size = 10000
input_size = 3  # (motion (float), color (float), context (bool/int))
output_size = 2  # (-1, 1) one-hot encode


def prune_network(i, j):
    FILE = f"mante_nets/mante_rnn_{i}_pruned_{j}.pth"
    print(f'start prune{i}_{j}')
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size).to(device)
    model.load_state_dict(torch.load(f"mante_nets/mante_rnn_{i}.pth", map_location=device))
    model.rnn.all_weights[0][1][j][j] *= 0  # [0] to squeeze, [1] for hh weights, [to_node], [from_node]
    torch.save(model.state_dict(), FILE)


def test_network(i, j):
    print(f'start test{i}_{j}')
    FILE = f"mante_nets/mante_rnn_{i}_pruned_{j}.pth"
    model = RNN(input_size, hidden_size, num_layers, output_size, batch_size).to(device)
    model.load_state_dict(torch.load(FILE, map_location=device))
    test_rnn(model, i, j)


def test_rnn(net, i, j):
    print(f'start analyze{i}_{j}')
    mc_dataset = MotionColorDataset(10000, 10)  # TODO pass dataset from training
    X, Y, Z, true_labels, C = mc_dataset.get_xyz(10000, context_time='retro', vary_acc=True)
    X_test = np.array(X)  # input
    U_test = np.array(true_labels)  # true label
    Y_test = np.array(Y)
    Z_test = np.array(Z)
    C_test = np.array(C)
    with torch.no_grad():
        num_test = 10000
        net.eval()
        hidden = net.init_hidden()
        output, hidden = net(torch.from_numpy(X_test).float(), hidden.float())
        Yhat = np.squeeze(output.numpy())
        predictions = (Yhat[:, 0] < Yhat[:, 1]).astype(int)  # Logic for 1-hot encoding of 0/1 at output node
        correct = (predictions == U_test).sum()
        accuracy = correct / num_test

        motion_indices = [i for i, e in enumerate(C_test) if e == -1]
        color_indices = [i for i, e in enumerate(C_test) if e == 1]
        acc_motion = sum([(U_test[i] == predictions[i]).astype(int) for i in motion_indices])/len(motion_indices)
        acc_color = sum([(U_test[i] == predictions[i]).astype(int) for i in color_indices])/len(color_indices)

    with open(f"mante_nets/mante_rnn_{i}_pruned_{j}_analysis.txt", "a") as f:
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"acc_motion: {acc_motion}\n")
        f.write(f"acc_color: {acc_color}\n")


def prune(i):
    for j in range(4):
        prune_network(i, j)
        test_network(i, j)


if __name__ == '__main__':
    with Pool(8) as p:
        p.map(prune, range(50))
