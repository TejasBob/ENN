import copy
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import os
from sklearn.metrics import roc_auc_score
import time
from torch.distributions.dirichlet import Dirichlet
from torch.autograd import Variable
import pickle
from torchvision import models

kl_loss = nn.KLDivLoss(reduction="batchmean")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_mean = [0.4914, 0.4822, 0.4465]
dataset_std = [0.24, 0.24, 0.26]

if not os.path.exists("model"):
	os.makedirs("model")

def relu_evidence(y):
    return F.relu(y)


# always with uncertainty
def get_prediction_and_uncertainty_for_single_image(model, img_tensor, device=None):
    if not device:
        device = get_device()
    num_classes = 10
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    img_variable = img_variable.to(device)

    
    output = model(img_variable)
    evidence = relu_evidence(output)
    alpha = evidence + 1
    uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
    _, preds = torch.max(output, 1)
    prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
    output = output.flatten()
    prob = prob.flatten()
    preds = preds.flatten()

    return preds[0].item(), uncertainty.item(), prob[0].item()

def create_figure2(model):
    dataset_name = "SVHN and CIFAR10"
    no_of_samples_from_each_dataset = 2500
    svhn_test_ = torchvision.datasets.SVHN('data/svhn', split="test", download=True,
                transform=transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize(mean=dataset_mean, std=dataset_std)]))
    svhn_test = torch.utils.data.Subset(svhn_test_, range(no_of_samples_from_each_dataset))
    print("Size of SVHN Test Set: ", len(svhn_test))

    cifar10_test = torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True,
                transform=transforms.Compose([transforms.ToTensor(), 
                transforms.Normalize(mean=dataset_mean, std=dataset_std)]))

    cifar10_test = torch.utils.data.Subset(cifar10_test, range(len(svhn_test)))

    print("Size of cropped Cifar10 Train Set: ", len(cifar10_test))
    dataset = torch.utils.data.ConcatDataset([svhn_test, cifar10_test])

    print("Size of Merged Test Set: ", len(dataset))
    labels_preds_uncertainties_probs = []
    for i, (img_tensor, label) in enumerate(dataset):
        pred, uncertainty, prediction_prob = get_prediction_and_uncertainty_for_single_image(model, img_tensor, device)
        if i < len(dataset) // 2:
            labels_preds_uncertainties_probs.append((label, pred, uncertainty, prediction_prob, 0))
        else: # CIFAR10 is ood
            labels_preds_uncertainties_probs.append((label, pred, uncertainty, prediction_prob, 1))

    thresholds = np.linspace(0, 1, 21) # 0, 0.05, 0.1 ... 0.95, 1
    results = []
    for thr in thresholds:
        temp = list(filter(lambda tup: tup[2] <= thr, labels_preds_uncertainties_probs))
        accuracy = 1 if len(temp) == 0 else len(list(filter(lambda tup: tup[0] == tup[1], temp))) / len(temp)
        results.append((thr, accuracy))

    x, y = zip(*results)
    plt.clf()
    fig_ = plt.gcf()
    plt.title(f"Trained and Tested using {dataset_name}")
    plt.xlabel("Uncertainty Threshold")
    plt.ylabel("Accuracy")
    plt.plot(x, y, "-s", c="k")
    """x0 = [2]
    y0 = [1]
    plt.plot(x0, y0, "o")"""

    fig_.savefig("figure2.jpg")
    # plt.show()
    
    with open('labels_preds_uncertainties_probs.pkl', 'wb') as f:
        pickle.dump(labels_preds_uncertainties_probs, f)


def one_hot_embedding(labels, num_classes=10, device="cpu"):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes, device=device)
    return y[labels]


def edl_loss(func, y, alpha, num_classes, kl_reg=True, device=None):
    # y = y.to(device)
    # alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    A_mean = torch.mean(A)

    if not kl_reg:
        return A_mean, 0

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_term = kl_divergence(kl_alpha, num_classes, device=device)
    kl_mean = torch.mean(kl_term)
    return A_mean, kl_mean


def KL_expectedProbSL_teacherProb(alpha, pretrainedProb, forward=True):
    S = torch.sum(alpha, dim=1, keepdims=True)
    prob = alpha / S
    if forward:
        kl = kl_loss(torch.log(prob), pretrainedProb)
    else:
        kl = kl_loss(torch.log(pretrainedProb), prob)
    return kl


# Uncertainty Cross Entropy (namely Expected Cross Entropy)
def edl_digamma_loss(
    output,
    target,
    epoch_num,
    num_classes,
    annealing_step,
    kl_lam,
    kl_lam_teacher,
    entropy_lam,
    pretrainedProb,
    forward,
    anneal=False,
    kl_reg=True,
    exp_type=5,
    device=None,
):
    if not device:
        device = set_device()
    if not kl_reg:
        assert anneal == False

    evidence = output
    alpha = evidence + 1

    ll_mean, kl_mean = edl_loss(
        torch.digamma, target, alpha, num_classes, kl_reg=kl_reg, device=device
    )

    if anneal:
        annealing_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32, device=device),
            torch.tensor(
                epoch_num / annealing_step, dtype=torch.float32, device=device
            ),
        )
        kl_div = annealing_coef * kl_mean
    else:
        kl_div = kl_lam * kl_mean

    # loss = ll_mean + kl_div

    #  KL divergence between
    # expected class probabilities and
    # the class probabilities predicted by detNN
    if exp_type == 3:
        kl = KL_expectedProbSL_teacherProb(alpha, pretrainedProb, forward=forward)
        loss = ll_mean + kl_div + kl_lam_teacher * kl
        return loss, ll_mean.detach(), kl_mean.detach(), kl.detach()

    # Entropy of Dirichlet as a regularizer
    if exp_type == 5:
        entropy = Dirichlet(alpha).entropy().mean()
        loss = ll_mean - entropy_lam * entropy
        return loss, ll_mean.detach(), entropy.detach()

    if exp_type == 6:
        ce = nll(alpha, target)
        entropy = Dirichlet(alpha).entropy().mean()
        loss = ce - entropy_lam * entropy
        return loss, ce.detach(), entropy.detach()



# ---------------- download data ---------------- 
dataset = torchvision.datasets.CIFAR10(
    root = './data/CIFAR10',
    train= True,
    transform=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.24, 0.24, 0.26])]),
    download = True,
)

val_size = 5000
train_size = len(dataset) - val_size
data_train, data_val = random_split(dataset, [train_size, val_size])
batch_size = 128
num_classes = 10

dataloader_train = DataLoader(data_train, batch_size, shuffle=True, num_workers=4, pin_memory=True)
dataloader_val = DataLoader(data_val, batch_size, num_workers=4, pin_memory=True)

print("traininig Set")
print("train data: ", len(dataloader_train))

print("testing Set")
print("testing data: ", len(dataloader_val))


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ResNet
        self.network = models.resnet18(pretrained=True)
        self.network.fc = torch.nn.Linear(512, num_classes) 
        self.output = nn.Softplus()

    def forward(self, x):
        logits = self.network(x)
        logits = self.output(logits)
        return logits


model = ResNet18(num_classes=num_classes)
model = model.to(device)
# summary(model, (3, 32, 32))

epochs = 20
lr = 1e-5
entropy_lam = 1e-1
weight_decay = 0.1
model_name = "./model/ResNet18_ENN_pretrained_epoch-{}_{}-{}_{}_{}.pt"

optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.1
    )

loss = edl_digamma_loss

print("device : ", device)

dataset_size_train = len(dataloader_train.dataset)
dataset_size_val = len(dataloader_val.dataset)
best_model_wts = copy.deepcopy(model.state_dict())

train_loss_arr = []
train_acc_arr = []

val_loss_arr = []
val_acc_arr = []

train_loss1_arr = []
train_loss2_arr = []
val_loss1_arr = []
val_loss2_arr = []

best_acc = 0.0
best_epoch = 0

running_loss = 0.0
best_model_epoch_no = 0

for epoch in range(epochs):
    start_time = time.time()
    running_loss = 0
    running_corrects = 0
    running_loss_1 = 0
    running_loss_2 = 0.0
    print("Epoch {}/{}".format(epoch+1, epochs))   
    print(f" get last lr:{scheduler.get_last_lr()}") if scheduler else "" 

    #  -----------  train part ----------
    model.train()
    for batch_idx, (inputs, labels) in enumerate(dataloader_train):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y = one_hot_embedding(labels, num_classes, device)
        train_loss, loss_first, loss_second = loss(
                outputs,
                y,
                epoch,
                num_classes,
                None,
                0,
                None,
                entropy_lam,
                None,
                None,
                kl_reg=False,
                exp_type=5,
                device=device,
            )

        train_loss.backward()
        optimizer.step()

        running_loss += train_loss.detach().item() * batch_size
        running_corrects += torch.sum(preds == labels).item()

        running_loss_1 += loss_first * batch_size
        running_loss_2 += loss_second * batch_size

    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / dataset_size_train
    epoch_acc = running_corrects / dataset_size_train

    epoch_loss_1 = running_loss_1 / dataset_size_train
    epoch_loss_2 = running_loss_2 / dataset_size_train

    train_loss_arr.append(round(epoch_loss, 4))
    train_acc_arr.append(round(epoch_acc, 4))
    train_loss1_arr.append(epoch_loss_1.item())
    train_loss2_arr.append(epoch_loss_2.item())

    print("Epoch " + str(epoch+1) + ":\n", 
         "Train : ", "loss: ", train_loss_arr[-1], ", loss1 : ", train_loss1_arr[-1],
         ", loss2_entropy : ", train_loss2_arr[-1],", accuracy : ", train_acc_arr[-1],  "\n")

	# ----------- save current model ----------- 
    torch.save(model.state_dict(), model_name.format(epoch, 32, 32, lr, entropy_lam))


	#  -----------  validation part ----------- 
    running_loss = 0
    running_corrects = 0
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    model.eval()
    for batch_idx, (inputs, labels) in enumerate(dataloader_val):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y = one_hot_embedding(labels, num_classes, device)

            val_loss, loss_first, loss_second = loss(
                outputs,
                y,
                epoch,
                num_classes,
                None,
                0,
                None,
                entropy_lam,
                None,
                None,
                kl_reg=False,
                exp_type=5,
                device=device,
            )
            running_loss += val_loss.detach().item() * batch_size
            running_corrects += torch.sum(preds == labels).item()

            running_loss_1 += loss_first * batch_size
            running_loss_2 += loss_second * batch_size

    epoch_loss = running_loss / dataset_size_val
    epoch_acc = running_corrects / dataset_size_val
    epoch_loss_1 = running_loss_1 / dataset_size_val
    epoch_loss_2 = running_loss_2 / dataset_size_val

    val_loss_arr.append(round(epoch_loss, 4))
    val_acc_arr.append(round(epoch_acc, 4))
    val_loss1_arr.append(epoch_loss_1.item())
    val_loss2_arr.append(epoch_loss_2.item())

    print("Epoch " + str(epoch+1) + ":\n", 
         "Val : ", "loss: ", val_loss_arr[-1], ", loss1 : ", val_loss1_arr[-1],
         ", loss2_entropy : ", val_loss2_arr[-1],", accuracy : ", val_acc_arr[-1],  "\n")

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_epoch = epoch
        best_model_epoch_no = epoch
        print(f"The best epoch: {best_epoch}, acc: {best_acc:.4f}.")
        best_model_wts = copy.deepcopy(model.state_dict())


    end_time = time.time()
    print("time for epoch : ", end_time - start_time)

plt.clf()
fig1 = plt.gcf()
plt.plot(list(range(epochs)), train_loss_arr, label='Training Loss')
plt.plot(list(range(epochs)), val_loss_arr, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Loss (Cross Entropy)')
plt.legend()
fig1.savefig("traininig_loss_cifar10_ENN_ResNet18_pretrained_{}-{}_{}.png".format(32,32,lr), bbox_inches='tight')

plt.clf()
fig2 = plt.gcf()
plt.plot(list(range(epochs)), train_acc_arr, label='Training Accuracy')
plt.plot(list(range(epochs)), val_acc_arr, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Average accuracy')
plt.legend()
fig2.savefig("traininig_acc_cifar10_ENN_ResNet18_pretrained_{}-{}_{}.png".format(32,32,lr), bbox_inches='tight')


plt.clf()
fig3 = plt.gcf()
plt.plot(list(range(epochs)), train_loss1_arr, label='Training loss1')
plt.plot(list(range(epochs)), val_loss1_arr, label='Validation loss1')
plt.xlabel('Epoch')
plt.ylabel('Loss1')
plt.legend()
fig3.savefig("traininig_loss1_cifar10_ENN_ResNet18_pretrained_{}-{}_{}.png".format(32,32,lr), bbox_inches='tight')

plt.clf()
fig4 = plt.gcf()
plt.plot(list(range(epochs)), train_loss2_arr, label='Training loss2_entropy')
plt.plot(list(range(epochs)), val_loss2_arr, label='Validation loss2_entropy')
plt.xlabel('Epoch')
plt.ylabel('Loss2_entropy')
plt.legend()
fig4.savefig("traininig_loss2_cifar10_ENN_ResNet18_pretrained_{}-{}_{}.png".format(32,32,lr), bbox_inches='tight')


# save model & load model
best_model_path = model_name.format(best_model_epoch_no, 32, 32, lr, entropy_lam)
print("loading best model : ", best_model_path)
best_model = ResNet18(num_classes=10)
best_model.load_state_dict(torch.load(best_model_path))
best_model = best_model.to(device)
best_model.eval()


prediction_arr = []
labels_arr = []

print("calculating AUC ROC score")
for batch_idx, (inputs, labels) in enumerate(dataloader_val):
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    with torch.no_grad():
        prob = best_model(inputs)
        prob = torch.nn.functional.softmax(prob, dim=1)
        tmp1 = labels.detach().cpu().numpy().tolist()
        tmp2 = prob.detach().cpu().numpy().tolist()
        labels_arr += tmp1
        prediction_arr += tmp2

print("roc_auc_score : ", roc_auc_score(labels_arr, prediction_arr, multi_class='ovr'))

print("generating figure2")
create_figure2(best_model)

with open('labels_preds_uncertainties_probs.pkl', 'rb') as f:
    labels_preds_uncertainties_probs = np.array(pickle.load(f))

from sklearn.metrics import roc_curve, auc
n_classes = 1

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
labels, predicted_classes, uncertainty_scores, prediction_probs, is_ood = labels_preds_uncertainties_probs.T
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true=is_ood, y_score=1-uncertainty_scores, pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.clf()
fig5 = plt.gcf()
ax = fig5.add_subplot(111)
# Plot of a ROC curve for a specific class
for i in range(n_classes):
    p = ax
    p.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    p.plot([0, 1], [0, 1], 'k--')
    p.set_xlim([0.0, 1.0])
    p.set_ylim([0.0, 1.05])
    p.set_xlabel('False Positive Rate')
    p.set_ylabel('True Positive Rate')
    p.set_title(f'ROC Curve for OOD (y=1) vs non-OOD (y=0)')
    p.legend(loc="lower right")

# fig5.show()
fig5.savefig("roc_curves_cifar10.png", bbox_inches='tight')
print("End")


