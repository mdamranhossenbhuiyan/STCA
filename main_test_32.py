from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from thop import profile
import pdb
import data_manager
from video_loader import VideoDataset
import transforms as T
import models
from models import resnet3d_mutual
from utils import AverageMeter, Logger, save_checkpoint,Cmc,np_cdist
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
import more_itertools as mit
import pdb
from reidtools import* 
parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
#parser.add_argument('-j', '--workers', default=0, type=int,
 #   choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=384,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=8, help="number of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max-epoch', default=450, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=40, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=200, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet503d_single_mid', help="resnet503d_single_mid,resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])
# Miscs
parser.add_argument('--print-freq', type=int, default=1, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--pretrained-model', type=str, default='~/log/checkpoint_ep150.pth.tar', help='need to be set for resnet3d models')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=50,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--use-cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu-devices', default='0,1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
   # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    transform_train = T.Compose([
        T.Random2DTranslation(args.height, args.width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    if args.arch=='resnet503d_single_mid':
        model = resnet3d_mutual.resnet50(num_classes=dataset.num_train_pids, sample_width=args.width, sample_height=args.height, sample_duration=args.seq_len)
        if not os.path.exists(args.pretrained_model):
            raise IOError("Can't find pretrained model: {}".format(args.pretrained_model))
        print("Loading checkpoint from '{}'".format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
 #       state_dict = {}
#        for key in checkpoint['state_dict']:
  #          if 'fc' in key: continue
   #         state_dict[key.partition("module.")[2]] = checkpoint['state_dict'][key]
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))


    input = torch.randn (1,1,3, 384, 128)
    flops, params = profile (model, inputs = (input,))    

    print(flops,params)



    if use_gpu:
        model = nn.DataParallel(model).cuda()

   # if args.evaluate:
    print("Evaluate only")
    rank, distmat = test(model, queryloader, galleryloader, args.pool, use_gpu)
#    visualize_ranked_results(distmat,dataset, data_type ='video',save_dir=args.save_dir)




def get_features(model,imgs,test_num_tracks):
    """to handle higher seq length videos due to OOM error
    specifically used during test
   
    Arguments:
        model -- model under test
        imgs -- imgs to get features for
   
    Returns:
        features
    """

    # handle chunked data
    all_features = []
    for test_imgs in mit.chunked(imgs, test_num_tracks):
        current_test_imgs = torch.stack(test_imgs)
        num_current_test_imgs = current_test_imgs.shape[0]
        outputs, features = model(current_test_imgs)
        features = features.view(num_current_test_imgs, -1)

        all_features.append(features)

    return torch.cat(all_features)

def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()

    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
        if use_gpu:
            imgs = imgs.cuda()
        
        with torch.no_grad():    
            imgs = Variable(imgs)
            b, n, s, c, h, w = imgs.size()
            imgs = imgs.view(b*n, s , c, h, w)
            assert(b==1)
          #  pdb.set_trace()
#            foo,features = model(imgs)
        
            #foo,features = model(imgs)
            features = get_features(model,imgs,32)   

            #features = features.view(n,-1)
            #if pool == 'avg':
            features = torch.mean(features, 0)
            #else:
             #   features, _ = torch.max(features, 0)
            torch.cuda.empty_cache()

        features = features.data.cpu()
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
        print(batch_idx,n,s)
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs.cuda()
        with torch.no_grad():    
            imgs = Variable(imgs)
            b, n, s, c, h, w = imgs.size()
            imgs = imgs.view(b*n, s , c, h, w)
            assert(b==1)
            #pdb.set_trace()
            #foo,features = model(imgs)
            features = get_features(model,imgs,32) 
            #features = features.view(n,-1)
            #if pool == 'avg':
            features = torch.mean(features, 0)
            #else:
             #   features, _ = torch.max(features, 0)
            torch.cuda.empty_cache()

        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
        print(batch_idx,n,s)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    qf = qf.numpy()
    gf = gf.numpy()

    distmat = np_cdist(qf,gf)




    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
#    CMC_or, MAP_OR = Cmc(qf,gf,q_pids, g_pids, q_camids, g_camids, 20)    
#    pdb.set_trace()
    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")



    return cmc[0],distmat

if __name__ == '__main__':
    main()





