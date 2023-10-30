import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from typing import Optional, Any, Tuple
import tqdm

from torch.autograd import Function


import torchvision
from torchvision import models
from torch.autograd import Variable


import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets import DatasetFolder
from PIL import Image
import os
import random
import argparse
import pickle

class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer

class ResNet34Fc(nn.Module):
    def __init__(self):
        super(ResNet34Fc, self).__init__()
        model_resnet34 = models.resnet34(pretrained=True)
        self.conv1 = model_resnet34.conv1
        self.bn1 = model_resnet34.bn1
        self.relu = model_resnet34.relu
        self.maxpool = model_resnet34.maxpool
        self.layer1 = model_resnet34.layer1
        self.layer2 = model_resnet34.layer2
        self.layer3 = model_resnet34.layer3
        self.layer4 = model_resnet34.layer4
        self.avgpool = model_resnet34.avgpool
        self.__in_features = model_resnet34.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Parameters:
            - **alpha** (float, optional): :math:`α`. Default: 1.0
            - **lo** (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            - **hi** (float, optional): Final value of :math:`\lambda`. Default: 1.0
            - **max_iters** (int, optional): :math:`N`. Default: 10
            - **auto_step** (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 10., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

class DANNNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=224, width=224, class_num=11):
        super(DANNNet, self).__init__()
        ## set base network
        self.base_network = ResNet34Fc()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=10, auto_step=True)
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, 1)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters, TCL does not train the base network.
        self.parameter_list = [{"params":self.bottleneck_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer.parameters(), "lr":1},
                               {"params":self.classifier_layer_2.parameters(), "lr":1}]

    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        outputs_adv = self.sigmoid(outputs_adv)
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv

class DANN(object):
    def __init__(self, base_net='ResNet50', width=224, class_num=11, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = DANNNet(base_net, use_bottleneck, width, width, class_num)
        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        domain_criterion = nn.BCELoss()
        _, outputs, softmax_outputs, outputs_adv = self.c_net(inputs)

        outputs_source = outputs.narrow(0, 0, labels_source.size(0))

        outputs_source_softmax = softmax_outputs.narrow(0, 0, labels_source.size(0))
        outputs_target_softmax = softmax_outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        d_source = outputs.narrow(0, 0, labels_source.size(0))
        num_classes = 11

        # Flatten spatial dimensions
        class_labels = labels_source.view(labels_source.size(0), -1)

        # Convert to long data type for scatter function
        class_labels = class_labels.long()

        # Create tensor for one-hot encoding
        one_hot = torch.zeros(class_labels.size(0), class_labels.size(1), num_classes, device=labels_source.device)

        # One-hot encoding
        one_hot.scatter_(2, class_labels.unsqueeze(-1), 1)

        class_labels = class_labels.view(class_labels.size(0), -1)  # Reshape to [4, 128*128]
        class_labels = torch.sum(class_labels, dim=1).long()  # Sum along the second dimension
        print("class_labels_indices",class_labels.shape) # torch.Size([4, 11])
        print("outputs_source",outputs_source.shape) # torch.Size([4, 11]) 
        classifier_loss = class_criterion(outputs_source, class_labels) # ERROR : RuntimeError: 0D or 1D target tensor expected, multi-target not supported


        source_domain_label = torch.FloatTensor(labels_source.size(0),1)
        target_domain_label = torch.FloatTensor(inputs.size(0) - labels_source.size(0),1)
        source_domain_label.fill_(1)
        target_domain_label.fill_(0)
        domain_label = torch.cat([source_domain_label,target_domain_label],0)
        domain_label = torch.autograd.Variable(domain_label.cuda())
        Ld = domain_criterion(outputs_adv, domain_label)

        self.iter_num += 1
        total_loss = classifier_loss + Ld
        return [total_loss, classifier_loss, Ld]

    def predict(self, inputs):
        feature, _, softmax_outputs,_= self.c_net(inputs)
        return softmax_outputs, feature

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

def get_data_paths(type_data):

    dataset_dir = "Dataset"  # Change this to the root directory of your dataset

    train_dir = os.path.join(dataset_dir, "train", type_data)
    test_dir = os.path.join(dataset_dir, "test", type_data)
    val_dir = os.path.join(dataset_dir, "val", type_data)

    # Initialize the mask-related variables to empty lists
    train_masks_file_paths = []
    test_masks_file_paths = []
    val_masks_file_paths = []

    if type_data =="images":
        masks_folder = "masks"  # Replace "images" with "masks" for mask data
    else:
        masks_folder = "noisy_masks"  # Replace "images" with "masks" for mask data
    train_masks_dir = os.path.join(dataset_dir, "train", masks_folder)
    test_masks_dir = os.path.join(dataset_dir, "test", masks_folder)
    val_masks_dir = os.path.join(dataset_dir, "val", masks_folder)

    # Get a list of file paths for masks
    train_masks_file_paths = [os.path.join(train_masks_dir, filename) for filename in os.listdir(train_masks_dir)]
    test_masks_file_paths = [os.path.join(test_masks_dir, filename) for filename in os.listdir(test_masks_dir)]
    val_masks_file_paths = [os.path.join(val_masks_dir, filename) for filename in os.listdir(val_masks_dir)]

    # Get a list of file paths in each directory
    train_file_paths = [os.path.join(train_dir, filename) for filename in os.listdir(train_dir)]
    test_file_paths = [os.path.join(test_dir, filename) for filename in os.listdir(test_dir)]
    val_file_paths = [os.path.join(val_dir, filename) for filename in os.listdir(val_dir)]

    # Combine all file paths into one list
    all_file_paths = train_file_paths + test_file_paths + val_file_paths
    all_masks_paths = train_masks_file_paths + test_masks_file_paths + val_masks_file_paths

    # Randomly shuffle the list
    random.shuffle(all_file_paths)
    percentage = 0.4
    if type_data == "images":
        percentage = 0.6

    # Choose a percentage of the file paths
    chosen_file_paths = all_file_paths[:int(percentage * len(all_file_paths))]  # used 40%
    chosen_masks_paths = all_masks_paths[:int(percentage * len(all_masks_paths))]
    
    #print(chosen_masks_paths)
    
    return chosen_file_paths, chosen_masks_paths
    
class CustomDataset(Dataset):
    def __init__(self, type_data, transform=None, loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.type_data=type_data
        self.image_paths,self.mask_paths = get_data_paths(self.type_data)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path =  self.image_paths[idx]
        image = Image.open(image_path)
        mask = Image.open(self.mask_paths[idx])


        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image,mask


def train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, iter_num, max_iter):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    labels_source_1D = labels_source
    print(inputs.shape,"torch.cat((inputs_source, inputs_target), dim=0)")
    total_loss = model_instance.get_loss(inputs, labels_source_1D)
    total_loss[0].backward()
    optimizer.step()

    return [total_loss[0].cpu().data.numpy(), total_loss[1].cpu().data.numpy(), total_loss[2].cpu().data.numpy()]


def evaluate(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data  # Assuming your data loader returns a tuple with inputs as the first element
        if model_instance.use_gpu:
            inputs = Variable(inputs).cuda()
        else:
            inputs = Variable(inputs)

        probabilities, feature = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        feature = feature.data.float()

        if first_test:
            all_probs = probabilities
            all_feature = feature
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_feature = torch.cat((all_feature, feature), 0)

    _, predict = torch.max(all_probs, 1)
    # Assuming labels are not available in your custom dataset
    # You can modify this part based on your data
    accuracy = None

    model_instance.set_train(ori_train_state)
    return {'accuracy': accuracy}, all_feature

def train(model_instance, source_dataloader, target_dataloader, test_target_loader, max_iter, optimizer, lr_scheduler, eval_interval,group_ratios):
    model_instance.set_train(True)
    print("start train...")
    loss = []  # accumulate total loss for visualization.
    result = []  # accumulate eval result on target data during training.
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    while True:
        for (datas_clean, datat) in tqdm.tqdm(
            zip(source_dataloader, target_dataloader),
            total=min(len(source_dataloader), len(target_dataloader)),
            desc='Train epoch = {}'.format(epoch), ncols=80, leave=False
        ):
            inputs_source, labels_source = datas_clean
            inputs_target, _ = datat

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num / 5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), Variable(labels_source).cuda()
            else:
                inputs_source, inputs_target, labels_source = Variable(inputs_source), Variable(inputs_target), Variable(labels_source)

            print(inputs_source.shape,"inputs_source")
            print(inputs_target.shape,"inputs_target")
            print(labels_source.shape,"labels_source")
            
            
            total_loss = train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer, iter_num, max_iter)

            # Validation
            if iter_num % eval_interval == 0 and iter_num != 0:
                eval_result, all_feature = evaluate(model_instance, test_target_loader)
                print('source domain:', eval_result)
                result.append(eval_result['accuracy'].cpu().data.numpy())

            iter_num += 1
            total_progress_bar.update(1)
            loss.append(total_loss)

        epoch += 1

        if iter_num > max_iter:
            # np.save('statistic/DANN_feature_target.npy', all_feature.cpu().numpy())
            break
    print('finish train')
    # torch.save(model_instance.c_net.state_dict(), 'statistic/DANN_model.pth')
    return [loss, result]



def main():
    
    stats_file = 'train_results.pkl'

    
    # Define your data transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create custom datasets for source and target domains
    source_dataset = CustomDataset(type_data="images", transform=data_transforms)
    target_dataset = CustomDataset(type_data="noisy_images", transform=data_transforms)

    # Create data loaders
    source_dataloader = DataLoader(source_dataset, batch_size=4, shuffle=True, num_workers=4)
    target_dataloader = DataLoader(target_dataset, batch_size=4, shuffle=True, num_workers=4)

    width = 224
    class_num = 11
    srcweight = 4
    model_instance = DANN(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight)
    param_groups = model_instance.get_parameter_list()

    # Initialize the variables directly
    sgd_params = {
        'lr': 0.01,  # Replace with your desired learning rate
        'momentum': 0.9,  # Replace with your desired momentum value
        'weight_decay': 1e-4,  # Replace with your desired weight decay value
        'nesterov': True  # Replace with your desired nesterov setting
    }
    group_ratios = [group['lr'] for group in param_groups]

    # Create the optimizer object with initialized variables
    optimizer = torch.optim.SGD(param_groups, **sgd_params)

    # Initialize the variables directly
    gamma = 0.9  # Replace with your desired value
    decay_rate = 0.5  # Replace with your desired value
    init_lr = 0.01  # Replace with your desired value
    # Create the lr_scheduler object with initialized variables
    lr_scheduler = INVScheduler(gamma=gamma, decay_rate=decay_rate, init_lr=init_lr)

    max_iter = 10  # Replace with the desired number of iterations
    eval_interval = 2  # Replace with the evaluation interval

    # Call the train function to start the training process
    train_results = train(model_instance, source_dataloader, target_dataloader, target_dataloader, max_iter, optimizer, lr_scheduler, eval_interval, group_ratios)
    
    # You can use train_results for further analysis or visualization
    # Check if stats_file is not None before saving
    if stats_file is not None:
        with open(stats_file, 'wb') as file:
            pickle.dump(train_results, file)
if __name__ == "__main__":
    main()