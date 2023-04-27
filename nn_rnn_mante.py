import copy
import time
import pickle

import torch
import torch.nn as nn
import numpy as np
import torchvision.models
from numpy import nonzero
# from torch.optim.lr_scheduler import MultiStepLR

from analyze_info_flow import test_rnn, analyze_info_flow_rnn
from datasets.MotionColorDataset import MotionColorDataset
import matplotlib.pyplot as plt
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10000  # 100,000 takes around 2 minutes w/ 1 layer hidden size 4, 1k batch size
learning_rate = 0.001  # .0001
hidden_size = 4
num_layers = 1
batch_size = 10000
input_size = 3  # (motion (float), color (float), context (bool/int))
output_size = 2  # (-1, 1) one-hot encode
debug = False

# Dataset params
# train_dataset = MotionColorDataset(100, 10)
# test_dataset = MotionColorDataset(100, 10)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, nonlinearity="relu")
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.activations = []

    def forward(self, x, hidden):
        x1, hidden_i = self.rnn(x, hidden)
        x2 = self.fc1(hidden_i)
        # hidden state is equivalent to activation of RNN layer, no need to pass hidden_i to activations
        self.activations = [x1, x2]
        return x2, hidden_i

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            return torch.zeros(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.float64).to(device)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float64).to(device)

    def get_weights(self):
        weights = []
        # all_weights returns a 4-component list:
        # ih weights, hh weights, ih bias, hh bias
        for layer in self.rnn.all_weights[0][0:2]:
            weights.append(np.array(layer.data))
        weights.append(getattr(self, 'fc1').weight.data.numpy())  # TODO MM update if multiple layers
        # weights.append([getattr(self, 'fc%d' % i).weight.data.numpy() for i in range(1, 3)])
        return weights


def translate_to_cel(label):
    return nonzero(label == 1.0)[0]


# FILE = "model.pth"


# TODO finetune these
def set_optimal_weights(model):
    with torch.no_grad():
        model.rnn.all_weights[0][0].data.fill_(0)
        model.rnn.all_weights[0][1].data.fill_(0)
        model.rnn.all_weights[0][2].data.fill_(0)
        model.rnn.all_weights[0][3].data.fill_(0)

        # ih
        model.rnn.all_weights[0][0][0][0] = .1  # [0] to squeeze, [0] for ih weights, [to_node], [from_node]
        model.rnn.all_weights[0][0][1][1] = .1
        model.rnn.all_weights[0][0][0][2] = 100
        model.rnn.all_weights[0][0][1][2] = -100

        # hh
        model.rnn.all_weights[0][1][0][0] = 1
        model.rnn.all_weights[0][1][1][1] = 1
        ## when training, model converges to have also -.15 on the cross weights ([0][1][0][1] and [1][0] = -.156

        # ho
        model.fc1.weight[0][0] = 1
        model.fc1.weight[1][0] = 1
        model.fc1.weight[0][1] = -1
        model.fc1.weight[1][1] = -1

        # bias ih
        model.rnn.bias_ih_l0[0] = 10
        model.rnn.bias_ih_l0[1] = 10
        # also has a very small hh hidden bias on both (-.1ish)

        # bias fc out
        model.fc1.bias[0] = -110
        model.fc1.bias[1] = 110


def train_new_rnn(model=None):
    preds = []
    accs = []
    opt_accs = []
    not_adhere_sample_means_list = []
    if model is None:
        model = RNN(input_size, hidden_size, num_layers, output_size, batch_size).to(device)

    # TODO MM
    # set_optimal_weights(model)
    # initial_weights = copy.deepcopy(model.rnn.all_weights)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = MultiStepLR(optimizer,
    #                     milestones=[1000, 3000, 5000],  # List of epoch indices
    #                     gamma =0.1)
    print('training model')
    time_start = time.perf_counter()
    model.train()
    mc_dataset = MotionColorDataset(batch_size, 10)
    X, _, _, true_labels, C = mc_dataset.get_xyz(batch_size, context_time="retro", vary_acc=True)
    X = np.array(X)
    Y = np.array(true_labels)
    correct_history = []
    adherences = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        hidden = model.init_hidden()
        output, hidden = model(torch.from_numpy(X).float(), hidden.float())
        loss = criterion(torch.squeeze(output), torch.from_numpy(Y))
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # training accuracy and adherence
        Yhat = np.squeeze(output.detach().numpy())
        predictions = (Yhat[:, 0] < Yhat[:, 1]).astype(int)  # Logic for 1-hot encoding of 0/1 at output node
        correct = (predictions == Y).sum()
        correct_history.append(correct)
        context_indices = [(0 if i == -1 else 1) for i in C]
        sample_means = [X[i, :, context_indices[i]].sum()/10 for i in range(len(X))]
        pred_match_sample = [(1 if ((sample_means[i] > 0 and predictions[i] == 1) or (sample_means[i] < 0 and predictions[i] == 0)) else 0) for i in range(len(predictions))]
        adherences.append(sum(pred_match_sample))

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item() * 1000:.6f}')
            print(f'Time elapsed: {time.perf_counter() - time_start:0.2f} seconds')
            model.eval()
            pred_match_true, acc, opt_acc, not_adhere_sample_means = test_rnn(model)
            preds.append(pred_match_true)
            accs.append(acc)
            opt_accs.append(opt_acc)
            not_adhere_sample_means_list.append(sum(not_adhere_sample_means)/len(not_adhere_sample_means))
            # print(pred_match_true/10000)
            print(acc)
            # early stopping
            if epoch > 200:
                if acc < accs[-2] and acc < accs[-3] and acc < accs[-4]:
                    break
            # print(scheduler.get_last_lr())
    model.eval()
    print(preds)
    print(accs)
    # print(initial_weights[0][0] - model.rnn.all_weights[0][0])
    return model, preds, accs, opt_accs, not_adhere_sample_means_list, correct_history, adherences




# model = torchvision.models.vgg16(weights='DEFAULT')


# model = RNN(input_size, hidden_size, num_layers, output_size, batch_size).to(device)
# FILE = 'model224mid.pth'
# if os.path.isfile(FILE):
#     model.load_state_dict(torch.load(FILE))
#
# z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, \
# c_info_flows_weighted, accuracy, acc_motion, acc_color, acc_context = \
#             analyze_info_flow_rnn(model, 'linear-svm', model_name='model221')


# FILE = 'rnn_mante_random.pth'
# model, preds, accs, opt_accs, not_adhere_sample_means_list, correct_history, adherences = train_new_rnn()
#
# torch.save(model.state_dict(), FILE)

# load rnn_mante_random
# FILE = 'rnn_mante_random.pth'
# rand_model = RNN(input_size, hidden_size, num_layers, output_size, batch_size).to(device)
# rand_model.load_state_dict(torch.load(FILE, map_location=device))
#
# FILE = 'rnn_mante_retro.pth'
# retro_model = RNN(input_size, hidden_size, num_layers, output_size, batch_size).to(device)
# retro_model.load_state_dict(torch.load(FILE, map_location=device))
#
# z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy, acc_motion, acc_color, acc_context = \
#             analyze_info_flow_rnn(rand_model, 'linear-svm', model_name='rand_model')
#
# z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy, acc_motion, acc_color, acc_context = \
#             analyze_info_flow_rnn(retro_model, 'linear-svm', model_name='retro_model')



# #
# # z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy, acc_motion, acc_color, acc_context = \
# #             analyze_info_flow_rnn(model, 'corr', model_name='raw_model')
#
# with torch.no_grad():
#     model.load_state_dict(torch.load(FILE))
#     model.rnn.all_weights[0][1][1][1] = 0
# z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy, acc_motion, acc_color, acc_context = \
#             analyze_info_flow_rnn(model, 'linear-svm', model_name='perturbed_model')
#
# train_new_rnn(model)
# z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy, acc_motion, acc_color, acc_context = \
#             analyze_info_flow_rnn(model, 'linear-svm', model_name='retrained_model')


# if os.path.isfile(FILE):
#     model.load_state_dict(torch.load(FILE))
# else:
# url = 'https://hooks.slack.com/services/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'
# for i in range(10):
#     model = train_new_rnn()
#     torch.save(model.state_dict(), 'model' + str(i) + '.pth')
#     acc = test_rnn(model)
#     myobj = {'text': str(acc)}
#     x = requests.post(url, json=myobj)
#     print(x.text)

# print(preds, accs)
#
# open_file = open("preds.pkl", "wb")
# pickle.dump(preds, open_file)
# open_file.close()
#
# open_file = open("accs.pkl", "wb")
# pickle.dump(accs, open_file)
# open_file.close()
#
# open_file = open("preds.pkl", "rb")
# preds = pickle.load(open_file)
# open_file.close()
#
# open_file = open("accs.pkl", "rb")
# accs = pickle.load(open_file)
# open_file.close()
#
# plt.plot([x/10000 for x in correct_history])
# plt.xlabel("Epoch")
# plt.ylabel("Training accuracy")
# plt.title("Training accuracy over training")
# plt.show()
#
# plt.plot([x/10000 for x in adherences])
# plt.xlabel("Epoch")
# plt.ylabel("Training adherence")
# plt.title("Training adherence over training")
# plt.show()

# plt.plot([x/10000 for x in preds])
# plt.xlabel("Epoch (thousands)")
# plt.ylabel("Predictions matching sign of sample mean (percent)")
# plt.title("Adherence to sample mean over training (10k epochs)")
# plt.show()
#
# plt.plot(accs)
# plt.plot(opt_accs)
# plt.legend(['accuracy achieved', 'optimal accuracy'], loc='upper left')
# plt.xlabel("Epoch (thousands)")
# plt.ylabel("Accuracy (percent)")
# plt.title("Accuracy over training (10k epochs)")
# plt.show()
# print("max preds: ", max(preds))
# print("max accs: ", max(accs))
#
# plt.plot(not_adhere_sample_means_list)
# plt.xlabel("Epoch (thousands)")
# plt.ylabel("Average sample mean when not adhering")
# plt.title("Average sample mean in non-adhere cases over training (10k epochs)")
# plt.show()


# print(acc_before_pruning)
# print(model.rnn.all_weights)
# z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy = \
#             analyze_info_flow_rnn(model, 'corr', model_name='model-pre-prune')
#
# with torch.no_grad():
#     model.rnn.all_weights[0][1][3][1] = 0  # prune the self hh weight for node bottom node from itself
#
# print(model.rnn.all_weights)
# z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy = \
#             analyze_info_flow_rnn(model, 'corr', model_name='model-post-prune-1')

# if os.path.isfile(FILE):
#     model.load_state_dict(torch.load(FILE))
# else:
# url = 'https://hooks.slack.com/services/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'
# for i in range(10):
#     model = train_new_rnn()
#     torch.save(model.state_dict(), 'model' + str(i) + '.pth')
#     acc = test_rnn(model)
#     myobj = {'text': str(acc)}
#     x = requests.post(url, json=myobj)
#     print(x.text)

# color_accs = []
# colors_accs_post = []
# motion_accs = []
# motion_accs_post = []
# context_accs = []
# context_accs_post = []
# for i in range(10):
#     model = RNN(input_size, hidden_size, num_layers, output_size, batch_size)
#     model.load_state_dict(torch.load('rnn-tests-22-11-15-4node/model'+str(i)+'.pth'))
#     print("testing " + str(i))
#     stdout = sys.stdout
#     with open('rnn-tests-22-11-16-4node-pruning/model'+str(i)+'.txt', 'w') as f:
#         sys.stdout = f
#         os.makedirs('rnn-tests-22-11-16-4node-pruning/model'+str(i))
#         z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy, acc_motion, acc_color, acc_context = \
#             analyze_info_flow_rnn(model, 'corr', model_name='model'+str(i))
#     with open('rnn-tests-22-11-16-4node-pruning/model' + str(i) + '-max-pruned.txt', 'w') as f:
#         sys.stdout = f
#         os.makedirs('rnn-tests-22-11-16-4node-pruning/model' + str(i) + '-max-pruned')
#         # TODO why y info flows? u flows maybe
#         j = np.where(z_info_flows[9] == max(z_info_flows[9]))[0][0]  # 9 is last layer before output
#         with torch.no_grad():
#             model.rnn.all_weights[0][1][j][j] = 0  #[0] to squeeze, [1] for hh weights, [to_node], [from_node]
#         z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after = \
#             analyze_info_flow_rnn(model, 'corr', model_name='model' + str(i) + '-max-pruned')
#         sys.stdout = stdout
#         motion_accs.append(acc_motion)
#         motion_accs_post.append(acc_motion_after)
#         color_accs.append(acc_color)
#         colors_accs_post.append(acc_color_after)
#         # plt.plot([acc_color, acc_color_after], [acc_motion, acc_motion_after])
#         # plt.show()
# plot_array_color = np.vstack((np.array(color_accs), np.array(colors_accs_post)))
# plot_array_motion = np.vstack((np.array(motion_accs), np.array(motion_accs_post)))
# plt.plot(plot_array_color, plot_array_motion, color='C0', alpha=0.3)
# plt.plot(plot_array_color.mean(axis=1), plot_array_motion.mean(axis=1), color='C0', linewidth=2)
# plt.show()
#
# model 0, seeing color after0_half = .5, after0 = 84, before=84
# accs = []
# for i in range(10):
#     x = []
#     model = RNN(input_size, hidden_size, num_layers, output_size, batch_size)
#     model.load_state_dict(torch.load('rnn-tests-22-11-15-4node/model0.pth'))
#     acc, motion, color_b, context = test_rnn(model)
#
#     model_0_pruned_half = copy.deepcopy(model)
#     with torch.no_grad():
#         model_0_pruned_half.rnn.all_weights[0][1][0][0] *= 0.5
#     acc, motion, color_ah, context = test_rnn(model_0_pruned_half)
#
#     model_0_pruned = copy.deepcopy(model)
#     with torch.no_grad():
#         model_0_pruned.rnn.all_weights[0][1][0][0] *= 0
#     acc, motion, color_a, context = test_rnn(model_0_pruned)
#
#     accs.append([color_b, color_ah, color_a])
# print(accs)



## RUN FULL ANALYSIS AND SAVE
# for i in range(10):
#     model = RNN(input_size, hidden_size, num_layers, output_size, batch_size)
#     model.load_state_dict(torch.load('rnn-tests-22-11-15-4node/model'+str(i)+'.pth'))
#
#     z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after = \
#         analyze_info_flow_rnn(model, 'corr', model_name='model' + str(i) + '-before')
#     joblib.dump([z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after], 'model-analysis'+str(i)+'-before')
#
#     prune_model = copy.deepcopy(model)
#     with torch.no_grad():
#         prune_model.rnn.all_weights[0][1][0][0] *= 0  # [0] to squeeze, [1] for hh weights, [to_node], [from_node]
#     z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after = \
#         analyze_info_flow_rnn(prune_model, 'corr', model_name='model' + str(i) + '-0-pruned')
#     joblib.dump(
#         [z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows,
#          c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after],
#         'model-analysis' + str(i) + '-after0')
#
#     prune_model = copy.deepcopy(model)
#     with torch.no_grad():
#         prune_model.rnn.all_weights[0][1][1][1] *= 0  # [0] to squeeze, [1] for hh weights, [to_node], [from_node]
#     z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after = \
#         analyze_info_flow_rnn(prune_model, 'corr', model_name='model' + str(i) + '-1-pruned')
#     joblib.dump(
#         [z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows,
#          c_info_flows_weighted, accuracy,
#          acc_motion_after, acc_color_after, acc_context_after], 'model-analysis' + str(i) + '-after1')
#
#     prune_model = copy.deepcopy(model)
#     with torch.no_grad():
#         prune_model.rnn.all_weights[0][1][2][2] *= 0  # [0] to squeeze, [1] for hh weights, [to_node], [from_node]
#     z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after = \
#         analyze_info_flow_rnn(prune_model, 'corr', model_name='model' + str(i) + '-2-pruned')
#     joblib.dump(
#         [z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows,
#          c_info_flows_weighted, accuracy,
#          acc_motion_after, acc_color_after, acc_context_after], 'model-analysis' + str(i) + '-after2')
#
#     prune_model = copy.deepcopy(model)
#     with torch.no_grad():
#         prune_model.rnn.all_weights[0][1][3][3] *= 0  # [0] to squeeze, [1] for hh weights, [to_node], [from_node]
#     z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after = \
#         analyze_info_flow_rnn(prune_model, 'corr', model_name='model' + str(i) + '-3-pruned')
#     joblib.dump(
#         [z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows,
#          c_info_flows_weighted, accuracy,
#          acc_motion_after, acc_color_after, acc_context_after], 'model-analysis' + str(i) + '-after3')
#
#     prune_model = copy.deepcopy(model)
#     with torch.no_grad():
#         prune_model.rnn.all_weights[0][1][0][0] *= 0.5  # [0] to squeeze, [1] for hh weights, [to_node], [from_node]
#     z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after = \
#         analyze_info_flow_rnn(prune_model, 'corr', model_name='model' + str(i) + '-0-pruned-half')
#     joblib.dump([z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after], 'model-analysis'+str(i)+'-after0-half')
#
#     prune_model = copy.deepcopy(model)
#     with torch.no_grad():
#         prune_model.rnn.all_weights[0][1][1][1] *= 0.5  # [0] to squeeze, [1] for hh weights, [to_node], [from_node]
#     z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after = \
#         analyze_info_flow_rnn(prune_model, 'corr', model_name='model' + str(i) + '-1-pruned-half')
#     joblib.dump([z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy,
#                  acc_motion_after, acc_color_after, acc_context_after], 'model-analysis' + str(i) + '-after1-half')
#
#     prune_model = copy.deepcopy(model)
#     with torch.no_grad():
#         prune_model.rnn.all_weights[0][1][2][2] *= 0.5  # [0] to squeeze, [1] for hh weights, [to_node], [from_node]
#     z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after = \
#         analyze_info_flow_rnn(prune_model, 'corr', model_name='model' + str(i) + '-2-pruned-half')
#     joblib.dump([z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy,
#                  acc_motion_after, acc_color_after, acc_context_after], 'model-analysis' + str(i) + '-after2-half')
#
#     prune_model = copy.deepcopy(model)
#     with torch.no_grad():
#         prune_model.rnn.all_weights[0][1][3][3] *= 0.5  # [0] to squeeze, [1] for hh weights, [to_node], [from_node]
#     z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy, acc_motion_after, acc_color_after, acc_context_after = \
#         analyze_info_flow_rnn(prune_model, 'corr', model_name='model' + str(i) + '-3-pruned-half')
#     joblib.dump([z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, c_mis, c_info_flows, c_info_flows_weighted, accuracy,
#                  acc_motion_after, acc_color_after, acc_context_after], 'model-analysis' + str(i) + '-after3-half')

# accs_motion_before = []
# accs_color_before = []
#
# accs_motion_after_max_motion = []
# accs_motion_after_max_color = []
# accs_color_after_max_motion = []
# accs_color_after_max_color = []
# accs_motion_after_min_motion = []
# accs_motion_after_min_color = []
# accs_color_after_min_motion = []
# accs_color_after_min_color = []
#
# accs_motion_after_max_motion_half = []
# accs_motion_after_max_color_half = []
# accs_color_after_max_motion_half = []
# accs_color_after_max_color_half = []
# accs_motion_after_min_motion_half = []
# accs_motion_after_min_color_half = []
# accs_color_after_min_motion_half = []
# accs_color_after_min_color_half = []

# MAX/MIN ANALYSIS
# for i in range(10):
#     print('start')
#     _,z_info_flows,_,_,y_info_flows,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis'+str(i)+'-before')
#     index_max_motion_flow = np.where(z_info_flows[9] == max(z_info_flows[9]))[0][0]  # 9 is last layer before output
#     index_max_color_flow = np.where(y_info_flows[9] == max(y_info_flows[9]))[0][0]
#     index_min_motion_flow = np.where(z_info_flows[9] == min(z_info_flows[9]))[0][0]
#     index_min_color_flow = np.where(y_info_flows[9] == min(y_info_flows[9]))[0][0]
#
#     _,_,_,_,_,_,_,acc_motion_after_max_color, acc_color_after_max_color, acc_context_after_max_color = joblib.load('model-analysis'+str(i)+'-after'+str(index_max_color_flow))
#     _,_,_,_,_,_,_,acc_motion_after_min_color, acc_color_after_min_color, acc_context_after_min_color = joblib.load('model-analysis'+str(i)+'-after'+str(index_min_color_flow))
#     _,_,_,_,_,_,_,acc_motion_after_max_motion, acc_color_after_max_motion, acc_context_after_max_motion = joblib.load('model-analysis'+str(i)+'-after'+str(index_max_motion_flow))
#     _,_,_,_,_,_,_,acc_motion_after_min_motion, acc_color_after_min_motion, acc_context_after_min_motion = joblib.load('model-analysis'+str(i)+'-after'+str(index_min_motion_flow))
#
#     accs_motion_before.append(acc_motion_before)
#     accs_color_before.append(acc_color_before)
#     accs_motion_after_max_motion.append(acc_motion_after_max_motion)
#     accs_motion_after_max_color.append(acc_motion_after_max_color)
#     accs_color_after_max_motion.append(acc_color_after_max_motion)
#     accs_color_after_max_color.append(acc_color_after_max_color)
#     accs_motion_after_min_motion.append(acc_motion_after_min_motion)
#     accs_motion_after_min_color.append(acc_motion_after_min_color)
#     accs_color_after_min_motion.append(acc_color_after_min_motion)
#     accs_color_after_min_color.append(acc_color_after_min_color)
#
#     _,_,_,_,_,_,_,acc_motion_after_max_color_half, acc_color_after_max_color_half, acc_context_after_max_color_half = joblib.load('model-analysis'+str(i)+'-after'+str(index_max_color_flow)+'-half')
#     _,_,_,_,_,_,_,acc_motion_after_min_color_half, acc_color_after_min_color_half, acc_context_after_min_color_half = joblib.load('model-analysis'+str(i)+'-after'+str(index_min_color_flow)+'-half')
#     _,_,_,_,_,_,_,acc_motion_after_max_motion_half, acc_color_after_max_motion_half, acc_context_after_max_motion_half = joblib.load('model-analysis'+str(i)+'-after'+str(index_max_motion_flow)+'-half')
#     _,_,_,_,_,_,_,acc_motion_after_min_motion_half, acc_color_after_min_motion_half, acc_context_after_min_motion_half = joblib.load('model-analysis'+str(i)+'-after'+str(index_min_motion_flow)+'-half')
#
#     accs_motion_after_max_motion_half.append(acc_motion_after_max_motion_half)
#     accs_motion_after_max_color_half.append(acc_motion_after_max_color_half)
#     accs_color_after_max_motion_half.append(acc_color_after_max_motion_half)
#     accs_color_after_max_color_half.append(acc_color_after_max_color_half)
#     accs_motion_after_min_motion_half.append(acc_motion_after_min_motion_half)
#     accs_motion_after_min_color_half.append(acc_motion_after_min_color_half)
#     accs_color_after_min_motion_half.append(acc_color_after_min_motion_half)
#     accs_color_after_min_color_half.append(acc_color_after_min_color_half)
#
# plot_array_color = np.vstack((np.array(accs_color_before), np.array(accs_color_after_max_motion_half), np.array(accs_color_after_max_motion)))
# print(plot_array_color)
# plot_array_motion = np.vstack((np.array(accs_motion_before), np.array(accs_motion_after_max_motion_half), np.array(accs_motion_after_max_motion)))
# print(plot_array_motion)
# plt.plot(plot_array_color, plot_array_motion, color='C3', alpha=0.3)
# plt.plot(plot_array_color.mean(axis=1), plot_array_motion.mean(axis=1), color='C3', linewidth=2, label='mean max motion')
# plt.xlabel("Color accuracy")
# plt.ylabel('Motion accuracy')
# ax = plt.gca()
# ax.set_title("Color and motion accuracy after pruning max motion edge")
# # plt.show()
#
# plot_array_color = np.vstack((np.array(accs_color_before), np.array(accs_color_after_max_color_half), np.array(accs_color_after_max_color)))
# plot_array_motion = np.vstack((np.array(accs_motion_before), np.array(accs_motion_after_max_color_half), np.array(accs_motion_after_max_color)))
# plt.plot(plot_array_color, plot_array_motion, color='C0', alpha=0.3)
# plt.xlabel("Color accuracy")
# plt.ylabel('Motion accuracy')
# plt.plot(plot_array_color.mean(axis=1), plot_array_motion.mean(axis=1), color='C0', linewidth=2, label='mean max color')
# ax = plt.gca()
# ax.set_title("Color and motion accuracy after pruning max color edge")
# # plt.show()
#
# plot_array_color = np.vstack((np.array(accs_color_before), np.array(accs_color_after_min_motion_half), np.array(accs_color_after_min_motion)))
# plot_array_motion = np.vstack((np.array(accs_motion_before), np.array(accs_motion_after_min_motion_half), np.array(accs_motion_after_min_motion)))
# plt.plot(plot_array_color, plot_array_motion, color='C1', alpha=0.3)
# plt.xlabel("Color accuracy")
# plt.ylabel('Motion accuracy')
# plt.plot(plot_array_color.mean(axis=1), plot_array_motion.mean(axis=1), color='C1', linewidth=2, label='mean min motion')
# ax = plt.gca()
# ax.set_title("Color and motion accuracy after pruning min motion edge")
# # plt.show()
#
# plot_array_color = np.vstack((np.array(accs_color_before), np.array(accs_color_after_min_color_half), np.array(accs_color_after_min_color)))
# plot_array_motion = np.vstack((np.array(accs_motion_before), np.array(accs_motion_after_min_color_half), np.array(accs_motion_after_min_color)))
# plt.plot(plot_array_color, plot_array_motion, color='C2', alpha=0.3)
# plt.xlabel("Color accuracy")
# plt.ylabel('Motion accuracy')
# plt.plot(plot_array_color.mean(axis=1), plot_array_motion.mean(axis=1), color='C2', linewidth=2, label='mean min color')
# ax = plt.gca()
# # ax.set_title("Color and motion accuracy after pruning min color edge")
# ax.set_title("Color and motion accuracy")
# leg = plt.legend(loc='upper center')
# plt.show()



# accs_motion_before = []
# accs_color_before = []
# accs_context_before = []
# accs_motion_after0 = []
# accs_color_after0 = []
# accs_context_after0 = []
# accs_motion_after1 = []
# accs_color_after1 = []
# accs_context_after1 = []
# accs_motion_after2 = []
# accs_color_after2 = []
# accs_context_after2 = []
# accs_motion_after3 = []
# accs_color_after3 = []
# accs_context_after3 = []
#
#
# for i in range(10):
#     _,_,_,_,_,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis'+str(i)+'-before')
#     _,_,_,_,_,_,_,acc_motion_after0, acc_color_after0, acc_context_after0 = joblib.load('model-analysis'+str(i)+'-after0')
#     _,_,_,_,_,_,_,acc_motion_after1, acc_color_after1, acc_context_after1 = joblib.load('model-analysis'+str(i)+'-after1')
#     _,_,_,_,_,_,_,acc_motion_after2, acc_color_after2, acc_context_after2 = joblib.load('model-analysis'+str(i)+'-after2')
#     _,_,_,_,_,_,_,acc_motion_after3, acc_color_after3, acc_context_after3 = joblib.load('model-analysis'+str(i)+'-after3')
#     _,_,_,_,_,_,_,accs_motion_before.append(acc_context_before)
#     _,_,_,_,_,_,_,accs_color_before.append(acc_color_before)
#     _,_,_,_,_,_,_,accs_context_before.append(acc_context_before)
#
#     accs_motion_after0.append(acc_motion_after0)
#     accs_color_after0.append(acc_color_after0)
#     accs_context_after0.append(acc_context_after0)
#
#     accs_motion_after1.append(acc_motion_after1)
#     accs_color_after1.append(acc_color_after1)
#     accs_context_after1.append(acc_context_after1)
#
#     accs_motion_after2.append(acc_motion_after2)
#     accs_color_after2.append(acc_color_after2)
#     accs_context_after2.append(acc_context_after2)
#
#     accs_motion_after3.append(acc_motion_after3)
#     accs_color_after3.append(acc_color_after3)
#     accs_context_after3.append(acc_context_after3)
#
# plot_array_color = np.vstack((np.array(accs_color_before), np.array(accs_color_after0)))
# plot_array_motion = np.vstack((np.array(accs_motion_before), np.array(accs_motion_after0)))
# plt.plot(plot_array_color, plot_array_motion, color='C0', alpha=0.3)
# plt.xlabel("Color accuracy")
# plt.ylabel('Motion accuracy')
# plt.plot(plot_array_color.mean(axis=1), plot_array_motion.mean(axis=1), color='C0', linewidth=2)
# ax = plt.gca()
# ax.set_title("Color and motion accuracy after pruning self-edge 0")
# plt.show()

# GRAPH
# accs_motion_before = []
# accs_color_before = []
# accs_motion_after0 = []
# accs_motion_after1 = []
# accs_motion_after2 = []
# accs_motion_after3 = []
# accs_color_after0 = []
# accs_color_after1 = []
# accs_color_after2 = []
# accs_color_after3 = []
# accs_motion_after0_half = []
# accs_motion_after1_half = []
# accs_motion_after2_half = []
# accs_motion_after3_half = []
# accs_color_after0_half = []
# accs_color_after1_half = []
# accs_color_after2_half = []
# accs_color_after3_half = []
# color_flow_by_edge = []
# motion_flow_by_edge = []
# context_flow_by_edge = []
# z_flows = []
# y_flows = []
#
# d = {}
# for i in range(10):
#     # Z is motion
#     # Y iz color
#     print('start')
#     _,z_info_flows,_,_,y_info_flows,_,_, c_info_flows, _,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis'+str(i)+'-before')
#     _,z_info_flows_after_0,_,_,y_info_flows_after_0,_,_,_,_,_,acc_motion_after_0, acc_color_after_0, _ = joblib.load('model-analysis'+str(i)+'-after'+str(0))
#     _,_,_,_,_,_,_,_,_,_,acc_motion_after_0_half, acc_color_after_0_half, _ = joblib.load('model-analysis'+str(i)+'-after'+str(0)+'-half')
#     _,z_info_flows_after_1,_,_,y_info_flows_after_1,_,_,_,_,_,acc_motion_after_1, acc_color_after_1, _ = joblib.load('model-analysis'+str(i)+'-after'+str(1))
#     _,_,_,_,_,_,_,_,_,_,acc_motion_after_1_half, acc_color_after_1_half, _ = joblib.load('model-analysis'+str(i)+'-after'+str(1)+'-half')
#     _,z_info_flows_after_2,_,_,y_info_flows_after_2,_,_,_,_,_,acc_motion_after_2, acc_color_after_2, _ = joblib.load('model-analysis'+str(i)+'-after'+str(2))
#     _,_,_,_,_,_,_,_,_,_,acc_motion_after_2_half, acc_color_after_2_half, _ = joblib.load('model-analysis'+str(i)+'-after'+str(2)+'-half')
#     _,z_info_flows_after_3,_,_,y_info_flows_after_3,_,_,_,_,_,acc_motion_after_3, acc_color_after_3, _ = joblib.load('model-analysis'+str(i)+'-after'+str(3))
#     _,_,_,_,_,_,_,_,_,_,acc_motion_after_3_half, acc_color_after_3_half, _ = joblib.load('model-analysis'+str(i)+'-after'+str(3)+'-half')
#
#     # _, z_info_flows, _, _, y_info_flows, _,  _,_,_,_, acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis' + str(i) + '-before')
#     #
#     # _, z_info_flows_after_0, _, _, y_info_flows_after_0, _, _,_,_, _, acc_motion_after_0, acc_color_after_0, _ = joblib.load(
#     #     'model-analysis' + str(i) + '-after' + str(0))
#     # _, _, _, _, _, _, _,_,_, _, acc_motion_after_0_half, acc_color_after_0_half, _ = joblib.load(
#     #     'model-analysis' + str(i) + '-after' + str(0) + '-half')
#     # _, z_info_flows_after_1, _, _, y_info_flows_after_1, _,  _,_,_,_, acc_motion_after_1, acc_color_after_1, _ = joblib.load(
#     #     'model-analysis' + str(i) + '-after' + str(1))
#     # _, _, _, _, _, _,  _,_,_,_, acc_motion_after_1_half, acc_color_after_1_half, _ = joblib.load(
#     #     'model-analysis' + str(i) + '-after' + str(1) + '-half')
#     # _, z_info_flows_after_2, _, _, y_info_flows_after_2, _,  _,_,_,_, acc_motion_after_2, acc_color_after_2, _ = joblib.load(
#     #     'model-analysis' + str(i) + '-after' + str(2))
#     # _, _, _, _, _, _, _,_,_, _, acc_motion_after_2_half, acc_color_after_2_half, _ = joblib.load(
#     #     'model-analysis' + str(i) + '-after' + str(2) + '-half')
#     # _, z_info_flows_after_3, _, _, y_info_flows_after_3, _, _,_,_, _, acc_motion_after_3, acc_color_after_3, _ = joblib.load(
#     #     'model-analysis' + str(i) + '-after' + str(3))
#     # _, _, _, _, _, _, _,_,_,_, acc_motion_after_3_half, acc_color_after_3_half, _ = joblib.load(
#     #     'model-analysis' + str(i) + '-after' + str(3) + '-half')
#
#     accs_motion_before.append(acc_motion_before)
#     accs_color_before.append(acc_color_before)
#     accs_motion_after0.append(acc_motion_after_0)
#     accs_motion_after1.append(acc_motion_after_1)
#     accs_motion_after2.append(acc_motion_after_2)
#     accs_motion_after3.append(acc_motion_after_3)
#     accs_color_after0.append(acc_color_after_0)
#     accs_color_after1.append(acc_color_after_1)
#     accs_color_after2.append(acc_color_after_2)
#     accs_color_after3.append(acc_color_after_3)
#     accs_motion_after0_half.append(acc_motion_after_0_half)
#     accs_motion_after1_half.append(acc_motion_after_1_half)
#     accs_motion_after2_half.append(acc_motion_after_2_half)
#     accs_motion_after3_half.append(acc_motion_after_3_half)
#     accs_color_after0_half.append(acc_color_after_0_half)
#     accs_color_after1_half.append(acc_color_after_1_half)
#     accs_color_after2_half.append(acc_color_after_2_half)
#     accs_color_after3_half.append(acc_color_after_3_half)
#     color_flow_by_edge.append(y_info_flows[9])
#     motion_flow_by_edge.append(z_info_flows[9])
#     z_flows.append(z_info_flows)
#     y_flows.append(y_info_flows)
#     # # context_flow_by_edge.append(c_info_flows[9])
#
#     d['model'+str(i)] = {'motion_flows': z_info_flows,
#                             'color_flows': y_info_flows,
#                             'acc_motion_before': acc_motion_before,
#                             'acc_color_before': acc_color_before,
#                             'acc_motion_after_0': acc_motion_after_0,
#                             'acc_color_after_0': acc_color_after_0,
#                             'acc_motion_after_0_half': acc_motion_after_0_half,
#                             'acc_color_after_0_half': acc_color_after_0_half,
#                             'acc_motion_after_1': acc_motion_after_1,
#                             'acc_color_after_1': acc_color_after_1,
#                             'acc_motion_after_1_half': acc_motion_after_1_half,
#                             'acc_color_after_1_half': acc_color_after_1_half,
#                             'acc_motion_after_2': acc_motion_after_2,
#                             'acc_color_after_2': acc_color_after_2,
#                             'acc_motion_after_2_half': acc_motion_after_2_half,
#                             'acc_color_after_2_half': acc_color_after_2_half,
#                             'acc_motion_after_3': acc_motion_after_3,
#                             'acc_color_after_3': acc_color_after_3,
#                             'acc_motion_after_3_half': acc_motion_after_3_half,
#                             'acc_color_after_3_half': acc_color_after_3_half,
#                             }
# df = pd.DataFrame(data=d)
# # joblib.dump(df, 'model_analysis_df')
# # df.plot.bar(x=df.columns, y='acc_motion_before')
#
# # color_alphas_0 = [i[0]/max_color_flow for i in color_flow_by_edge]
# # color_alphas_1 = [i[1]/max_color_flow for i in color_flow_by_edge]
# # color_alphas_2 = [i[2]/max_color_flow for i in color_flow_by_edge]
# # color_alphas_3 = [i[3]/max_color_flow for i in color_flow_by_edge]
# #
# # motion_alphas_0 = [i[0]/max_motion_flow for i in motion_flow_by_edge]
# # motion_alphas_1 = [i[1]/max_motion_flow for i in motion_flow_by_edge]
# # motion_alphas_2 = [i[2]/max_motion_flow for i in motion_flow_by_edge]
# # motion_alphas_3 = [i[3]/max_motion_flow for i in motion_flow_by_edge]
#
# # context_alphas_0 = [i[0]/max_context_flow for i in context_flow_by_edge]
# # context_alphas_1 = [i[1]/max_context_flow for i in context_flow_by_edge]
# # context_alphas_2 = [i[2]/max_context_flow for i in context_flow_by_edge]
# # context_alphas_3 = [i[3]/max_context_flow for i in context_flow_by_edge]
#
#
# color_alphas_0 = np.array([i[0] for i in color_flow_by_edge])
# color_alphas_1 = np.array([i[1] for i in color_flow_by_edge])
# color_alphas_2 = np.array([i[2] for i in color_flow_by_edge])
# color_alphas_3 = np.array([i[3] for i in color_flow_by_edge])
#
# # color_alphas_0 = color_alphas_0/np.linalg.norm(color_alphas_0)
# # color_alphas_1 = color_alphas_1/np.linalg.norm(color_alphas_1)
# # color_alphas_2 = color_alphas_2/np.linalg.norm(color_alphas_2)
# # color_alphas_3 = color_alphas_3/np.linalg.norm(color_alphas_3)
#
# motion_alphas_0 = np.array([i[0] for i in motion_flow_by_edge])
# motion_alphas_1 = np.array([i[1] for i in motion_flow_by_edge])
# motion_alphas_2 = np.array([i[2] for i in motion_flow_by_edge])
# motion_alphas_3 = np.array([i[3] for i in motion_flow_by_edge])
#
# # motion_alphas_0 = motion_alphas_0/np.linalg.norm(motion_alphas_0)
# # motion_alphas_1 = motion_alphas_1/np.linalg.norm(motion_alphas_1)
# # motion_alphas_2 = motion_alphas_2/np.linalg.norm(motion_alphas_2)
# # motion_alphas_3 = motion_alphas_3/np.linalg.norm(motion_alphas_3)
#
# # Z is motion
# cmap = plt.colormaps['RdBu']
# norm = plt.Normalize(0, 1.8)
# for i in range(10):
#     z_info_flows = z_flows[i]
#     y_info_flows = y_flows[i]
#     for h in range(9): # h is layer
#         x = norm(z_info_flows[h][0])
#         y = norm(y_info_flows[h][0])
#         c = np.arctan(y/x) if x != 0 else np.arctan(y/0.0001)
#         plt.scatter(x, y, color=cmap(norm(c)))
#         # plt.scatter(x, y, color=cmap(h/9))
#
#         x = norm(z_info_flows[h][1])
#         y = norm(y_info_flows[h][1])
#         c = np.arctan(y/x) if x != 0 else np.arctan(y/0.0001)
#         plt.scatter(x, y, color=cmap(norm(c)))
#         # plt.scatter(x, y, color=cmap(h/9))
#
#         x = norm(z_info_flows[h][2])
#         y = norm(y_info_flows[h][2])
#         c = np.arctan(y/x) if x != 0 else np.arctan(y/0.0001)
#         plt.scatter(x, y, color=cmap(norm(c)))
#         # plt.scatter(x, y, color=cmap(h/9))
#
#         x = norm(z_info_flows[h][3])
#         y = norm(y_info_flows[h][3])
#         c = np.arctan(y/x) if x != 0 else np.arctan(y/0.0001)
#         plt.scatter(x, y, color=cmap(norm(c)))
#         # plt.scatter(x, y, color=cmap(h/9))
#
# plt.xlabel("Motion flow", fontsize=20)
# plt.ylabel('Color flow', fontsize=20)
# ax = plt.gca()
# ax.plot([0,1], [0,1], '--k', transform=ax.transAxes)
# ax.set_title("Motion and color flow by edge", fontsize=20)
# ax.tick_params(axis='both', which='major', labelsize=16)
# ax.tick_params(axis='both', which='minor', labelsize=14)
# plt.show()

# ratios0 = np.arctan(np.array([color_alphas_0[i]/motion_alphas_0[i] for i in range(len(color_alphas_0))]))
# min_ratio0 = min(ratios0)
# max_ratio0 = max(ratios0)
# ratios1 = np.arctan(np.array([color_alphas_1[i]/motion_alphas_1[i] for i in range(len(color_alphas_1))]))
# min_ratio1 = min(ratios1)
# max_ratio1 = max(ratios1)
# ratios2 = np.arctan(np.array([color_alphas_2[i]/motion_alphas_2[i] for i in range(len(color_alphas_2))]))
# min_ratio2 = min(ratios2)
# max_ratio2 = max(ratios2)
# ratios3 = np.arctan(np.array([color_alphas_3[i]/motion_alphas_3[i] for i in range(len(color_alphas_3))]))
# min_ratio3 = min(ratios3)
# max_ratio3 = max(ratios3)
# #
# min_ratio = min(min_ratio0, min_ratio1, min_ratio2, min_ratio3)
# max_ratio = max(max_ratio0, max_ratio1, max_ratio2, max_ratio3)

# softmax_ratios0 = np.tanh([(1 if np.argmax([color_alphas_0[i], motion_alphas_0[i]]) else -1) * max(scipy.special.softmax([color_alphas_0[i], motion_alphas_0[i]])) for i in range(len(color_alphas_0))])
# softmax_ratios1 = np.tanh([(1 if np.argmax([color_alphas_1[i], motion_alphas_1[i]]) else -1) * max(torch.softmax(torch.tensor([color_alphas_1[i], motion_alphas_1[i]]), 0)).item() for i in range(len(color_alphas_1))])
# softmax_ratios2 = np.tanh([(1 if np.argmax([color_alphas_2[i], motion_alphas_2[i]]) else -1) * max(torch.softmax(torch.tensor([color_alphas_2[i], motion_alphas_2[i]]), 0)).item() for i in range(len(color_alphas_2))])
# softmax_ratios3 = np.tanh([(1 if np.argmax([color_alphas_3[i], motion_alphas_3[i]]) else -1) * max(torch.softmax(torch.tensor([color_alphas_3[i], motion_alphas_3[i]]), 0)).item() for i in range(len(color_alphas_3))])
# min_soft_0 = min(softmax_ratios0)
# min_soft_1 = min(softmax_ratios1)
# min_soft_2 = min(softmax_ratios2)
# min_soft_3 = min(softmax_ratios3)
#
# max_soft_0 = max(softmax_ratios0)
# max_soft_1 = max(softmax_ratios1)
# max_soft_2 = max(softmax_ratios2)
# max_soft_3 = max(softmax_ratios3)
#
# min_soft = min([min_soft_0, min_soft_1, min_soft_2, min_soft_3])
# max_soft = max([max_soft_0, max_soft_1, max_soft_2, max_soft_3])


# These are not softmax ratios
# softmax_ratios0 = [color_alphas_0[i] - motion_alphas_0[i] for i in range(len(color_alphas_0))]
# softmax_ratios1 = [color_alphas_1[i] - motion_alphas_1[i] for i in range(len(color_alphas_1))]
# softmax_ratios2 = [color_alphas_2[i] - motion_alphas_2[i] for i in range(len(color_alphas_2))]
# softmax_ratios3 = [color_alphas_3[i] - motion_alphas_3[i] for i in range(len(color_alphas_3))]
#
# softmax_ratios0 = softmax_ratios0/np.linalg.norm(softmax_ratios0)
# softmax_ratios1 = softmax_ratios1/np.linalg.norm(softmax_ratios1)
# softmax_ratios2 = softmax_ratios2/np.linalg.norm(softmax_ratios2)
# softmax_ratios3 = softmax_ratios3/np.linalg.norm(softmax_ratios3)
#
# min_soft_0 = min(softmax_ratios0)
# min_soft_1 = min(softmax_ratios1)
# min_soft_2 = min(softmax_ratios2)
# min_soft_3 = min(softmax_ratios3)
#
# max_soft_0 = max(softmax_ratios0)
# max_soft_1 = max(softmax_ratios1)
# max_soft_2 = max(softmax_ratios2)
# max_soft_3 = max(softmax_ratios3)
#
# min_soft = min([min_soft_0, min_soft_1, min_soft_2, min_soft_3])
# max_soft = max([max_soft_0, max_soft_1, max_soft_2, max_soft_3])
#
# cmap = plt.colormaps['coolwarm_r']
# # norm = plt.Normalize(min_color, max_color)
# norm = plt.Normalize(min_soft, max_soft)
#
# for i in range(len(accs_color_before)):
#     x = [accs_color_before[i], accs_color_after0_half[i], accs_color_after0[i]]
#     y = [accs_motion_before[i], accs_motion_after0_half[i], accs_motion_after0[i]]
#     plt.plot(x, y, color=cmap(norm(softmax_ratios0[i])))
#     # plt.plot(x, y, color=cmap(norm(color_alphas_0[i])))
#     # plt.plot(x, y, color=cmap(softmax_ratios0[i]))
#
#     x = [accs_color_before[i], accs_color_after1_half[i], accs_color_after1[i]]
#     y = [accs_motion_before[i], accs_motion_after1_half[i], accs_motion_after1[i]]
#     plt.plot(x, y, color=cmap(norm(softmax_ratios1[i])))
#     # plt.plot(x, y, color=cmap(norm(color_alphas_1[i])))
#
#     x = [accs_color_before[i], accs_color_after2_half[i], accs_color_after2[i]]
#     y = [accs_motion_before[i], accs_motion_after2_half[i], accs_motion_after2[i]]
#     plt.plot(x, y, color=cmap(norm(softmax_ratios2[i])))
#     # plt.plot(x, y, color=cmap(norm(color_alphas_2[i])))
#
#     x = [accs_color_before[i], accs_color_after3_half[i], accs_color_after3[i]]
#     y = [accs_motion_before[i], accs_motion_after3_half[i], accs_motion_after3[i]]
#     plt.plot(x, y, color=cmap(norm(softmax_ratios3[i])))
#     # plt.plot(x, y, color=cmap(norm(color_alphas_3[i])))
#
#
# plt.xlabel("Color accuracy")
# plt.ylabel('Motion accuracy')
# ax = plt.gca()
# ax.set_title("Effect of pruning self-edge on motion and color accuracy (gradient: normed flow difference)")
# plt.colorbar(label="Like/Dislike Ratio", orientation="horizontal")
# plt.show()

# norm = plt.Normalize(min_color, max_color)

# ACCURACY REDUCTION TO FLOW
# for i in range(len(accs_motion_before)):
#     x = accs_motion_before[i] - accs_motion_after0[i]  # reduction in motion accuracy
#     # y = norm(motion_alphas_0[i])  # motion flow
#     y = motion_alphas_0[i]  # motion flow
#     plt.scatter(x, y, color=cmap(norm(motion_alphas_0[i])))
#
#     x = accs_motion_before[i] - accs_motion_after1[i]  # reduction in motion accuracy
#     # y = norm(motion_alphas_1[i])  # motion flow
#     y = motion_alphas_1[i]  # motion flow
#
#     plt.scatter(x, y, color=cmap(norm(motion_alphas_1[i])))
#
#     x = accs_motion_before[i] - accs_motion_after2[i]  # reduction in motion accuracy
#     # y = norm(motion_alphas_2[i])  # motion flow
#     y = motion_alphas_2[i]  # motion flow
#     plt.scatter(x, y, color=cmap(norm(motion_alphas_2[i])))
#
#     x = accs_motion_before[i] - accs_motion_after3[i]  # reduction in motion accuracy
#     # y = norm(motion_alphas_3[i])  # motion flow
#     y = motion_alphas_3[i]  # motion flow
#     plt.scatter(x, y, color=cmap(norm(motion_alphas_3[i])))
#
#
# plt.xlabel("Reduction in motion accuracy")
# plt.ylabel('Motion flow')
# ax = plt.gca()
# ax.set_title("Reduction in motion accuracy by motion flow on output edge")
# plt.show()

# cmap = plt.colormaps['jet']
#
# for i in range(len(accs_color_before)):
#     x = [accs_color_before[i], accs_color_after0_half[i], accs_color_after0[i]]
#     y = [accs_motion_before[i], accs_motion_after0_half[i], accs_motion_after0[i]]
#     plt.plot(x, y, color=cmap(motion_alphas_0[i]))
#
#     x = [accs_color_before[i], accs_color_after1_half[i], accs_color_after1[i]]
#     y = [accs_motion_before[i], accs_motion_after1_half[i], accs_motion_after1[i]]
#     plt.plot(x, y, color=cmap(motion_alphas_1[i]))
#
#     x = [accs_color_before[i], accs_color_after2_half[i], accs_color_after2[i]]
#     y = [accs_motion_before[i], accs_motion_after2_half[i], accs_motion_after2[i]]
#     plt.plot(x, y, color=cmap(motion_alphas_2[i]))
#
#     x = [accs_color_before[i], accs_color_after3_half[i], accs_color_after3[i]]
#     y = [accs_motion_before[i], accs_motion_after3_half[i], accs_motion_after3[i]]
#     plt.plot(x, y, color=cmap(motion_alphas_3[i]))
#
# plt.xlabel("Color accuracy")
# plt.ylabel('Motion accuracy')
# ax = plt.gca()
# ax.set_title("Effect of pruning self-edge on motion and color accuracy (gradient: motion)")
# plt.show()

# cmap = plt.colormaps['Greens']

# for i in range(len(accs_color_before)):
#     x = [accs_color_before[i], accs_color_after0_half[i], accs_color_after0[i]]
#     y = [accs_motion_before[i], accs_motion_after0_half[i], accs_motion_after0[i]]
#     plt.plot(x, y, color=cmap(context_alphas_0[i]))
#
#     x = [accs_color_before[i], accs_color_after1_half[i], accs_color_after1[i]]
#     y = [accs_motion_before[i], accs_motion_after1_half[i], accs_motion_after1[i]]
#     plt.plot(x, y, color=cmap(context_alphas_1[i]))
#
#     x = [accs_color_before[i], accs_color_after2_half[i], accs_color_after2[i]]
#     y = [accs_motion_before[i], accs_motion_after2_half[i], accs_motion_after2[i]]
#     plt.plot(x, y, color=cmap(context_alphas_2[i]))
#
#     x = [accs_color_before[i], accs_color_after3_half[i], accs_color_after3[i]]
#     y = [accs_motion_before[i], accs_motion_after3_half[i], accs_motion_after3[i]]
#     plt.plot(x, y, color=cmap(context_alphas_3[i]))
#
# plt.xlabel("Color accuracy")
# plt.ylabel('Motion accuracy')
# ax = plt.gca()
# ax.set_title("Effect of pruning self-edge on motion and color accuracy (gradient: context)")
# plt.show()

    # x = [accs_color_before[i], accs_color_after1_half[i], accs_color_after1]
    # plt.plot(accs_color_before[i], accs_color_after1_half[i], accs_color_after1, color=cmap(color_alphas_1[i]), alpha=0.3)
    # plt.plot(accs_color_before[i], accs_color_after2_half[i], accs_color_after2, color=cmap(color_alphas_2[i]), alpha=0.3)
    # plt.plot(accs_color_before[i], accs_color_after3_half[i], accs_color_after3, color=cmap(color_alphas_3[i]), alpha=0.3)

# for i in range(len(accs_motion_before)):
#     plt.plot(accs_motion_before[i], accs_motion_after0_half[i], accs_motion_after0, color=cmap(motion_alphas_0[i]), alpha=0.3)
#     plt.plot(accs_motion_before[i], accs_motion_after1_half[i], accs_motion_after1, color=cmap(motion_alphas_1[i]), alpha=0.3)
#     plt.plot(accs_motion_before[i], accs_motion_after2_half[i], accs_motion_after2, color=cmap(motion_alphas_2[i]), alpha=0.3)
#     plt.plot(accs_motion_before[i], accs_motion_after3_half[i], accs_motion_after3, color=cmap(motion_alphas_3[i]), alpha=0.3)

# plot_array_color = np.vstack((np.array(accs_color_before), np.array(accs_color_after0_half), np.array(accs_color_after0)))
# plot_array_motion = np.vstack((np.array(accs_motion_before), np.array(accs_motion_after0_half), np.array(accs_motion_after0)))
# plt.plot(plot_array_color, plot_array_motion, color=cmap(color_alphas_0), alpha=0.3, label='0 edge')
# # plt.plot(plot_array_color.mean(axis=1), plot_array_motion.mean(axis=1), color='C3', linewidth=2, label='0 edge')
# # plt.show()
#
# plot_array_color = np.vstack((np.array(accs_color_before), np.array(accs_color_after1_half), np.array(accs_color_after1)))
# plot_array_motion = np.vstack((np.array(accs_motion_before), np.array(accs_motion_after1_half), np.array(accs_motion_after1)))
# plt.plot(plot_array_color, plot_array_motion, color=cmap(color_alphas_1), alpha=0.3, label='1 edge')
# # plt.plot(plot_array_color.mean(axis=1), plot_array_motion.mean(axis=1), color='C0', linewidth=2, label='1 edge')
#
# plot_array_color = np.vstack((np.array(accs_color_before), np.array(accs_color_after2_half), np.array(accs_color_after2)))
# plot_array_motion = np.vstack((np.array(accs_motion_before), np.array(accs_motion_after2_half), np.array(accs_motion_after2)))
# plt.plot(plot_array_color, plot_array_motion, color=cmap(color_alphas_2), alpha=0.3, label='2 edge')
# # plt.plot(plot_array_color.mean(axis=1), plot_array_motion.mean(axis=1), color='C1', linewidth=2, label='2 edge')
#
# plot_array_color = np.vstack((np.array(accs_color_before), np.array(accs_color_after3_half), np.array(accs_color_after3)))
# plot_array_motion = np.vstack((np.array(accs_motion_before), np.array(accs_motion_after3_half), np.array(accs_motion_after3)))
# plt.plot(plot_array_color, plot_array_motion, color=cmap(color_alphas_3), alpha=0.3, label='3 edge')
# plt.plot(plot_array_color.mean(axis=1), plot_array_motion.mean(axis=1), color='C2', linewidth=2, label='3 edge')
# leg = plt.legend(loc='upper center')
# ax = plt.gca()
# ax.set_title("Effect of pruning self-edge on motion and color accuracy")
# plt.show()

# z_mis, z_info_flows, z_info_flows_weighted, y_mis, y_info_flows, y_info_flows_weighted, accuracy = \
#     analyze_info_flow_rnn(model, 'corr')
    # analyze_info_flow_rnn(model, 'linear-svm')  # use corr for estimate via correlation

# _,z_info_flows,_,_,y_info_flows,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis9-before')
# print("before " + acc_motion_before, acc_color_before, acc_context_before)
#
# _,z_info_flows,_,_,y_info_flows,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis9-after0')
# print(acc_motion_before, acc_color_before, acc_context_before)
#
# _,z_info_flows,_,_,y_info_flows,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis9-after1')
# print(acc_motion_before, acc_color_before, acc_context_before)
#
# _,z_info_flows,_,_,y_info_flows,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis9-after2')
# print(acc_motion_before, acc_color_before, acc_context_before)
#
# _,z_info_flows,_,_,y_info_flows,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis9-after3')
# print(acc_motion_before, acc_color_before, acc_context_before)
#
# _,z_info_flows,_,_,y_info_flows,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis9-after0-half')
# print(acc_motion_before, acc_color_before, acc_context_before)
#
# _,z_info_flows,_,_,y_info_flows,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis9-after1-half')
# print(acc_motion_before, acc_color_before, acc_context_before)
#
# _,z_info_flows,_,_,y_info_flows,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis9-after2-half')
# print(acc_motion_before, acc_color_before, acc_context_before)
#
# _,z_info_flows,_,_,y_info_flows,_,_,acc_motion_before, acc_color_before, acc_context_before = joblib.load('model-analysis9-after3-half')
# print(acc_motion_before, acc_color_before, acc_context_before)
#

