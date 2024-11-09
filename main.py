import argparse
import os
import random
import shutil
import time
import warnings
import sys
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from imgnet_models.resnet_gate import my_resnet50, my_resnet101, my_resnet34
from imgnet_models.mobilenetv2_custom import my_mobilenet_v2
from imgnet_models.vae import VAE
from imgnet_models.hypernet import HyperStructure
from Warmup_Sch import GradualWarmupScheduler
from alignment_functions import SelectionBasedRegularization, SelectionBasedRegularization_MobileNet, SelectionBasedRegularization_MobileNetV3
from torch.utils.data.dataset import random_split

from repeat_dataloader import MultiEpochsDataLoader
from utils import *
from train import *


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
                    # choices=model_names,
                    # help='model architecture: ' +
                    #     ' | '.join(model_names) +
                    #     ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lmd', default=10, type=float, metavar='W', help='group lasso lamd (default: 10)',
                    dest='lmd')

parser.add_argument('--epsilon', default=0.1, type=float, metavar='M',
                    help='epsilon in OTO')

parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--p', default=0.5, type=float,
                    help='Pruning Rate')
parser.add_argument('--stage', default='train-gate', type=str,
                    help='Which stage to choose')
parser.add_argument('--ls', default=True, type=str2bool)
parser.add_argument('--mix_up', default=False, type=str2bool)

parser.add_argument('--mbnet', default=False, type=str2bool)

# parser.add_argument('--distill', default=False, type=str2bool)
parser.add_argument('--gates', default=2, type=int)
parser.add_argument('--pruning_method', default='flops', type=str)

parser.add_argument('--base', default=3.0, type=float)
parser.add_argument('--interval', default=30, type=int)
parser.add_argument('--base_p', default=1.0, type=float)
parser.add_argument('--scratch', type=str2bool, default=False)

parser.add_argument('--bn_decay',type=str2bool, default=False)
parser.add_argument('--cos_anneal',type=str2bool, default=False)
parser.add_argument('--opt_name',type=str,default='SGD')

parser.add_argument('--project',type=str,default='gl')

parser.add_argument('--hyper_step', default=20, type=int)

parser.add_argument('--grad_mul', default=10.0, type=float)
parser.add_argument('--reg_w', default=4.0, type=float)  # 4.0 
parser.add_argument('--gl_lam', default=0.0001, type=float, help='group lasso lamda (default: 0.0001)')
parser.add_argument('--start_epoch_hyper', default=20, type=int)
parser.add_argument('--start_epoch_gl', default=100, type=int)

parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')
parser.add_argument('--vae', default=False, type=bool,help='vae model')

#/data/ILSVRC2012
best_acc1 = 0
def main():
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        print("Use GPU: {} for training".format(args.gpu))

    print(args)
    global best_acc1

    print("=> creating model '{}'".format(args.arch))
    
    print("args.state:", args.stage)
    if args.stage == 'train-gate':
        if args.arch == 'resnet50':
            if args.gates == 2:
                gate_string = '_2gates'
            else:
                gate_string = ''

            # state_dict = torch.load('./checkpoint/%s_base%s.pt'%(args.arch, gate_string))
            # model.load_state_dict(state_dict['state_dict'])
            if args.pretrained:
                print(">>>>>>>>> Using Pretrained Model <<<<<<<<<<<<<")
                model = my_resnet50(pretrained=True)
            else:
                print(">>>>>>>>> NO Pretrained Model <<<<<<<<<<<<<")
                model = my_resnet50()
                
            print(model)
            print("ResNet50 model structure")
            args.model_name = 'resnet'
            args.block_string = model.block_string

            print_model_param_flops(model) # Number of FLOPs: 4.12283G
                                                               #[64] * 6 . [128] * 8 . [256] * 12 . [512] * 6  virtual_gate number
            width, structure = model.count_structure() # 32, [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512]

            hyper_net = HyperStructure(structure=structure, T=0.4, base=3,args=args) # do we need to change HyperStructure? self.Bi_GRU = nn.GRU(64, 128, bidirectional=True)
            hyper_net.cuda(args.gpu)
            tmp = hyper_net()
            print("Mask and its size:", tmp, tmp.size()) # size:  torch.Size([7552])
            input('check mask size')
            import pdb; pdb.set_trace()
            # return

            args.structure = structure
            sel_reg = SelectionBasedRegularization(args) # Perform projection operator and update following Eq. (2) or Eq. (3) on ZIGs with w

            if args.pruning_method == 'flops': # default is flops
                size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnetbb(model)
                resource_reg = Flops_constraint_resnet_bb(args.p, size_kernel, size_out, size_group, size_inchannel,
                                                       size_outchannel, w=args.reg_w, HN=True,structure=structure)
                # eq(4) ?
        elif args.arch == 'vae':
            model = VAE()
            
            # get the model structure
            
            # init the HyperStructure with the model structure
            
            # init the regularizer
            
            # init the  sel_reg(apply to net) and resource_reg(apply to hypernet)

            if args.pretrained:
                print(">>>>>>>>> Using Pretrained Model <<<<<<<<<<<<<")
                model = VAE(pretrained=True)
            else:
                print(">>>>>>>>> NO Pretrained Model <<<<<<<<<<<<<")
                model = VAE()
                
            print(model)
            print("VAE model structure")
            args.model_name = 'vae'
            args.block_string = model.block_string

            import pdb; pdb.set_trace()
            print_model_param_flops(model, input_res = 128) # Number of FLOPs, vae has input_res = 128
            width, structure = model.count_structure() # 

            hyper_net = HyperStructure(structure=structure, T=0.4, base=3,args=args) # do we need to change HyperStructure? self.Bi_GRU = nn.GRU(64, 128, bidirectional=True)
            print('device:', args.gpu) # device: 7
            hyper_net.cuda(args.gpu)
            tmp = hyper_net()
            print("Mask and its size:", tmp, tmp.size()) # size:  torch.Size([7552])
            
            
            

            args.structure = structure
            sel_reg = SelectionBasedRegularization(args) # Perform projection operator and update following Eq. (2) or Eq. (3) on ZIGs with w

            if args.pruning_method == 'flops': # default is flops
                # size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_vae(model)
                resource_reg =  Channel_constraint(args.p) # use channel constraint for vae 
            
            
        elif args.arch == 'resnet101':
            if args.gates == 2:
                gate_string = '_2gates'
            else:
                gate_string = ''
            # args.model_name = 'resnet'
            # state_dict = torch.load('./checkpoint/%s_base%s.pt' % (args.arch, gate_string))
            # model.load_state_dict(state_dict['state_dict'])
            model = my_resnet101()
            args.model_name = 'resnet'
            args.block_string = model.block_string

            width, structure = model.count_structure()

            hyper_net = HyperStructure(structure=structure, T=0.4, base=3,args=args)

            hyper_net.cuda()
            print_model_param_flops(model)

            args.structure = structure
            sel_reg = SelectionBasedRegularization(args)

            size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnetbb(model)
            resource_reg = Flops_constraint_resnet_bb(args.p, size_kernel, size_out, size_group, size_inchannel,
                                     
                                                      size_outchannel, w=args.reg_w, HN=True,structure=structure)
        elif args.arch == 'resnet34':

            if args.gates == 2:
                gate_string = '_2gates'
            else:
                gate_string = ''
            print("ResNet34 ready")
            model = my_resnet34(num_gate=1)
            args.model_name = 'resnet'
            args.block_string = model.block_string

            print_model_param_flops(model)

            width, structure = model.count_structure()
            
            hyper_net = HyperStructure(structure=structure, T=0.4, base=3,args=args)
            hyper_net.cuda(args.gpu)

            args.structure = structure
            sel_reg = SelectionBasedRegularization(args)

            size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnet(model)
            resource_reg = Flops_constraint_resnet(args.p, size_kernel, size_out, size_group, size_inchannel,
                                                      size_outchannel, w=args.reg_w, HN=True,structure=structure)


# 0.31413G
# 0.27539G
# https://github.com/d-li14/mobilenetv2.pytorch/tree/master
        elif args.arch == 'mobnetv2':
            model = my_mobilenet_v2(gate_flag=True)

            if args.pretrained:
                model_p = my_mobilenet_v2(gate_flag=False)
                model_p.load_state_dict(torch.load('mobilenetv2_1.0-0c6065bc.pth'), strict=False)

                model_ms = list(model.modules())
                model_P_ms = list(model_p.modules())

                for m in model_ms:
                    if isinstance(m, virtual_gate):
                        model_ms.remove(m)

                for layer_id in range(len(model_ms)):
                    m0 = model_P_ms[layer_id]
                    m1 = model_ms[layer_id]
                    if isinstance(m0, nn.BatchNorm2d):
                        m1.weight.data.copy_(m0.weight.data.clone())
                        m1.bias.data.copy_(m0.bias.data.clone())
                        m1.running_mean.copy_(m0.running_mean.clone())
                        m1.running_var.copy_(m0.running_var.clone())
                    elif isinstance(m0, nn.Conv2d):
                        m1.weight.data.copy_(m0.weight.data.clone())
                    elif isinstance(m0, nn.Linear):
                        m1.weight.data.copy_(m0.weight.data.clone())
                        m1.bias.data.copy_(m0.bias.data.clone())

                del model_p
                print("pretrain mobnetv2 loaded")

            print("mobnetv2 ready")

            p = args.p
            args.model_name = 'mobnetv2'

            print_model_param_flops(model)

            width, structure = model.count_structure()

            args.structure = structure
            hyper_net = HyperStructure(structure=structure, T=0.4,base=args.base, args=args)
            sel_reg = SelectionBasedRegularization_MobileNet(args)

            hyper_net.cuda()
            size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_mobnet(
                model, input_res=224)
            resource_reg = Flops_constraint_mobnet(p, size_kernel, size_out, size_group, size_inchannel,
                                                      size_outchannel, w=args.reg_w, HN=True,structure=structure)

        args.selection_reg = sel_reg
        args.resource_constraint = resource_reg

    elif args.stage == 'baseline':
        if args.arch == 'resnet50':
            model = my_resnet50()


    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss() #.cuda(args.gpu)
    if args.stage == 'train-gate':
        params_group = group_weight(hyper_net) 
        print("len(list(hyper_net.parameters())): ", len(list(hyper_net.parameters())))

        optimizer_hyper = torch.optim.AdamW(params_group, lr=1e-3, weight_decay=1e-2)
        scheduler_hyper = torch.optim.lr_scheduler.MultiStepLR(optimizer_hyper, milestones=[int(0.98 * ((args.epochs - 5) / 2) + 5)], gamma=0.1)


        if args.bn_decay:
            print('bn decay')
            params = group_weight(model)
        else:
            print('bn no decay')
            params = model.parameters()
        opt_name = args.opt_name.lower() # 'adamw'
        model.lmbda_amplify   = DEFAULT_OPT_PARAMS[opt_name]['lmbda_amplify'] # 20
        model.hat_lmbda_coeff = DEFAULT_OPT_PARAMS[opt_name]['hat_lmbda_coeff'] # 1000.0
        model.lmd = args.lmd # group lasso lamd (default: 10) # 0.0

        print("====== model.lmd, group lasso lamd:", model.lmd)

        model.epsilon = args.epsilon

        if opt_name == 'sgd':
            optimizer = torch.optim.SGD(
                params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif opt_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(params, lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
        elif opt_name == 'adamw': # default 
            optimizer = torch.optim.AdamW(params, lr=args.lr,weight_decay=args.weight_decay)

        else:
            raise RuntimeError("Invalid optimizer {}. Only SGD and RMSprop are supported for gate training.".format(args.opt))
        print('optimizer:', optimizer)
        # torch.optim.lr_scheduler.CosineAnnealingLR is learning rate scheduler provided by PyTorch that adjusts the learning rate following a cosine annealing schedule
        if args.cos_anneal:
            # base_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.epochs - 5))
            base_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int((args.epochs - 5)/ 2)) # int((args.epochs - 5)/ 2) + 1) # int((240 - 5) / 2)
        else: # default
            base_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.epochs - 5))

        print("optimizer scheduler >>>", base_sch)

        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=base_sch)
        print("base_sch:", base_sch)
    else:
        if args.bn_decay:
            print('bn not decay')
            params = group_weight(model)
        else:
            print('bn decay')
            params = model.parameters()
        if args.opt_name == 'SGD':
            optimizer = torch.optim.SGD(params, args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt_name == 'ADAMW':
            optimizer = torch.optim.AdamW(params, args.lr,
                                          weight_decay=args.weight_decay)
        print(optimizer)


        if args.scratch:
            # base_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.epochs-5))
            base_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int((args.epochs - 5)/ 2) + 1) # int((240 - 5) / 2)

            print(">>>>>>>>>>>>>>>>>>>>>  ERROR <<<<<<<<<<<<<<<")
        else:
            if args.cos_anneal:
                print(">>>>>>>>>>>>>>>>>>>>>  ERROR <<<<<<<<<<<<<<<")

                # base_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(args.epochs - 5)) # COSINEANNEALINGWARMRESTARTS
                base_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int((args.epochs - 5)/ 2) + 1) # int((240 - 5) / 2)

            else:
                remain_epochs = args.epochs - 5
                drop_point = [int((remain_epochs - 10) / 3), int((remain_epochs - 10) / 3 * 2), int(remain_epochs - 10)]
                print("drop_point", drop_point)
                base_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_point, gamma=0.1)

        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=base_sch)

        print(base_sch)


    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    # traindir = os.path.join(args.data, 'val')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if  args.stage == 'train-gate':
        if args.mbnet:
            train_dataset = datasets.ImageFolder(
                    traindir,
                    transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]))
        elif args.vae:
            train_dataset = datasets.ImageFolder(
                    traindir,
                    transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        normalize,
                    ]))

        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.4,
                            hue=0.2),
                    normalize,
                ]))

    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    if not args.vae:
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    else: # vae
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                        transforms.Resize((128, 128)),
                        transforms.ToTensor(),
                        normalize,
            ])) 

    print("train_dataset:\n", train_dataset)
    print("val_dataset:\n", val_dataset)

    train_loader = MultiEpochsDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    if args.stage=='train-gate':

        ratio = (len(train_loader)/args.hyper_step)/len(train_loader)
        print("Val gate rate %.4f" % ratio)
        _, val_gate_dataset = random_split(
            train_dataset,
            lengths=[len(train_dataset) - int(ratio * len(train_dataset)), int(ratio * len(train_dataset))]
        )
        val_loader_gate = MultiEpochsDataLoader(
            val_gate_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,)
        # hyper_step

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    acc1 = 0
    swa_update = 0

    print('Label Smoothing: %s'%(args.ls))
    print_flops(hyper_net, args)
    cur_maskVec = None
    # print('Cutmix: %s'%(args.cutmix))
    for epoch in range(args.start_epoch, args.epochs): # default 0, 245
        #if args.stage != 'train-gate':
        for param_group in optimizer.param_groups:
            print("current_lr %.4f"%param_group["lr"])
            model.lr = param_group["lr"]

        # train for one epoch
        if args.stage == 'train-gate': # training start from here? train the main renset model and hypernet(controller network) at the same time?
            cur_maskVec = soft_train(train_loader, model, hyper_net, criterion, val_loader_gate, optimizer, optimizer_hyper, epoch, cur_maskVec, args) # 'epoch' of outter for is passed into soft_train
            scheduler.step()
            scheduler_hyper.step()

        elif args.stage == 'baseline':
            simple_train(train_loader, model, criterion, optimizer, epoch, args)
            scheduler.step()

        # print evaluate on validation set
        if args.stage == 'train-gate':
            if (epoch+1)%10 == 0:
                if epoch >= args.start_epoch_gl: # 50
                    acc1 = validateMask(val_loader, model, copy.deepcopy(cur_maskVec), criterion, args)
                else:
                    acc1 = validate(val_loader, model, criterion, args)
                print_flops(hyper_net, args)
            elif epoch >= int((args.epochs - 5) / 3 * 2) + 5:
                # if epoch >= args.start_epoch_gl:
                print("Testing masked")
                acc1 = validateMask(val_loader, model, copy.deepcopy(cur_maskVec), criterion, args)

                print("Testing ")
                acc1 = validate(val_loader, model, criterion, args)

                print_flops(hyper_net, args)

            elif epoch == 0:
                if args.vae:
                    print("skip validation for vae")
                else:
                    acc1 = validate(val_loader, model, criterion, args)
                    
                    print_flops(hyper_net, args)

            if hasattr(model, 'module'):
                model.module.reset_gates()
            else:
                model.reset_gates()

        else:
            acc1 = validate(val_loader, model, criterion, args)

############################
    print("Training Done")
    if isinstance(model, torch.nn.DataParallel):
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'hyper_net':hyper_net.state_dict(),
        }, args)
    else:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'hyper_net': hyper_net.state_dict(),
        }, args)
    print("Svaed !!!!")

def save_checkpoint(state, args, epochs=None):
    stage_string = ''
    if args.stage == 'train-gate':
        stage_string = 'gate'
    elif args.stage == 'baseline':
        stage_string = 'base'

    if args.arch == 'mobnetv2':
        arch_str = args.arch + '-'+ str(args.p)
    else:
        arch_str = args.arch

    if epochs is not None:
        epoch_string = str(epochs)
    else:
        epoch_string = ''

    import os
    os.makedirs('./checkpoint/', exist_ok=True)

    filename = './checkpoint/%s%s%s.pth.tar' % (arch_str+'_'+stage_string, str(args.lmd), epoch_string)
    torch.save(state, filename)


if __name__ == '__main__':
    main()