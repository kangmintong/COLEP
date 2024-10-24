from dataset.datasets_AwA2 import get_dataset
from torch.utils.data import DataLoader
from model.architectures_AwA2 import get_architecture
import torch
from torch.nn import  Sigmoid, Softmax
from tqdm import tqdm

batch_size = 50
noise_sd = 0.25

train_dataset = get_dataset('AWA', 'train')
test_dataset = get_dataset('AWA', 'test')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                          num_workers=8, pin_memory=False)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=8, pin_memory=False)

# model = get_architecture('resnet50', 'AWA')
# model_path = f'/data/common/AwA2/saved_models/noise_sd_{noise_sd}/main.pth.tar'
# checkpoint = torch.load(model_path, map_location='cpu')
# model.load_state_dict(checkpoint['state_dict'])
# model = model.cuda()
# model.eval()

all_path = [f'/data/common/AwA2/saved_models/noise_sd_{noise_sd:.2f}/main.pth.tar']
all_path.extend([f'/data/common/AwA2/saved_models/noise_sd_{noise_sd:.2f}/attribute_{i}.pth.tar' for i in range(85)])
all_path.extend([f'/data/common/AwA2/saved_models/noise_sd_{noise_sd:.2f}/hierarchy_{i}.pth.tar' for i in range(28)])


logits = []
labels = []

with torch.no_grad():
    print_freq = 10
    for i, (inputs, targets) in tqdm(enumerate(test_loader)):

        inputs, targets = inputs.cuda(), targets.cuda()
        cur_prediction = []
        labels.append(targets)
        for model_id, path in enumerate(all_path):
            checkpoint = torch.load(path, map_location='cpu')
            if model_id == 0:
                model = get_architecture('resnet50', 'AWA', classes = 50)
                m = Softmax(dim=1)
            else:
                model = get_architecture('resnet50', 'AWA', classes = 1)
                m = Sigmoid()
            model.load_state_dict(checkpoint['state_dict'])
            model = model.cuda()
            model.eval()

            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
            y_hat = m(model(inputs))
            cur_prediction.append(y_hat)
        cur_prediction = torch.concat(cur_prediction,dim=1)
        logits.append(cur_prediction)
        if i==2:
            break

logits = torch.concat(logits, dim=0)
print(f'logits.shape: {logits.shape}')



# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#         self.t = 0
#         self.n = 0
#         self.tp = 0
#         self.tn = 0
#         self.avg_tp = 0
#         self.avg_tn = 0
#     def update(self, val, n=1, pred = None, target = None):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#         if pred != None:
#             self.t += torch.sum(target[target != 0])
#             self.n += n - torch.sum(target[target != 0])
#             self.tp += torch.sum(pred[target != 0] != 0)
#             self.tn += torch.sum(pred[target == 0] == 0)
#             self.avg_tp = self.tp * 100.0 /self.t
#             self.avg_tn = self.tn * 100.0 /self.n
#
#
# top1 = AverageMeter()
# top5 = AverageMeter()
# with torch.no_grad():
#     print_freq = 10
#     for i, (inputs, targets) in tqdm(enumerate(test_loader)):
#
#         inputs, targets = inputs.cuda(), targets.cuda()
#
#         # augment inputs with noise
#         inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd
#
#         # compute output
#         outputs = model(inputs)
#
#
#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
#
#         top1.update(acc1.item(), inputs.size(0))
#         top5.update(acc5.item(), inputs.size(0))
#
#
#
#         if i % print_freq == 0:
#             print('Test: [{0}/{1}]\t'
#                   'Acc@1 {top1.avg:.3f}\t'
#                   'Acc@5 {top5.avg:.3f}'.format(
#                 i, len(test_loader), top1=top1, top5=top5))