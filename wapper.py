from tqdm import tqdm
import torch
from utils import set_seed, get_kernelsize, calculate_MI, reyi_entropy, hsic
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import warnings
import scipy.io as sio





warnings.filterwarnings("ignore")
set_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CE_loss = nn.CrossEntropyLoss()

def train_and_evaluate(args, param, trainloader, testloader, net):
    epochs = args.epochs
    learning_rate = args.lr
    gamma = args.gamma

    step_size = args.step_size
    schedulerisTrue = args.scheduler
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_acc = 0
    best_result = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0), disable=True):
            inputs, Sn_s, labels = data['views'], data['Sn'], data['label']
            inputs, Sn_s, labels = [input.to(torch.float32).to(device) for input in inputs], [Sn.to(torch.float32).to(device) for Sn in Sn_s], labels.to(device) 
            optimizer.zero_grad()
            inputs, com_feature, com_features, uni_features, outputs = net(inputs, Sn_s)
            fea_z = [torch.cat((uni_feature_i, com_feature),dim=1) for uni_feature_i in uni_features]
            sigma_com = get_kernelsize(com_feature, selected_param=0.15, select_type='1')
            sigma_un = [get_kernelsize(u, selected_param=0.15, select_type='1') for u in uni_features]
            sigma_input_list = [get_kernelsize(input, selected_param=0.15, select_type='1') for input in inputs]
            sigma_z_list = [get_kernelsize(z_i,selected_param=0.15, select_type='1') for z_i in fea_z]
            loss_ce = CE_loss(outputs, labels)
            I_xz = [calculate_MI(input_i, zi, s_x=sigma_input ** 2, s_y=sigma_z ** 2) for (input_i, zi, sigma_input, sigma_z) in zip(inputs, fea_z, sigma_input_list, sigma_z_list)]
            H_zc = reyi_entropy(com_feature, sigma_com)
            hsic_c_u = [hsic(com_feature, u_i, s_x=sigma_com ** 2, s_y=sigma_uni ** 2) for (u_i, sigma_uni) in zip(uni_features, sigma_un)]
            loss = loss_ce #+ param[0] * sum(I_xz) - param[1] * H_zc + param[2] * sum(hsic_c_u)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        if schedulerisTrue == args.scheduler:
            scheduler.step()     
        else:
            pass

        report = evaluate(args, testloader, net)
        net.train()
        # print(f'Epoch = {epoch}, best_acc = {best_acc}')
        if report[0] > best_acc:
            best_acc = report[0]
            best_result = report
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch = {epoch}, best_acc = {best_acc}, Current learning rate: {current_lr}')
    return best_result


def evaluate(args, testloader, net):
    net.eval()
    with torch.no_grad():
        correct = 0 
        total = 0
        y_true = []
        y_pred = []
        for i, data in tqdm(enumerate(testloader, 0), disable=True):
            inputs, Sn_s, labels = data['views'], data['Sn'], data['label']
            inputs, Sn_s, labels = [input.to(torch.float32).to(device) for input in inputs], [Sn.to(torch.float32).to(device) for Sn in Sn_s], labels.to(device) 
            
            _, _, _, _, outputs = net(inputs, Sn_s)
            _, predicted = torch.max(outputs.data, 1)

            y_true += labels.tolist()
            y_pred += predicted.tolist()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        acc, precision, recall, f1_score   = report['accuracy'], report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score']
        return [acc, precision, recall, f1_score]