import json

import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange, tqdm


class VideoDataset(Dataset):

    def get_vocab_size(self):
        return len(self.get_vocab())

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt, mode):
        super(VideoDataset, self).__init__()
        self.mode = mode  # to load train/val/test data

        # load the json file which contains information about the dataset
        self.captions = json.load(open(opt["caption_json"]))
        info = json.load(open(opt["info_json"]))
        self.ix_to_word = info['ix_to_word']
        self.word_to_ix = info['word_to_ix']
        print('vocab size is ', len(self.ix_to_word))
        self.splits = info['videos']
        print('number of train videos: ', len(self.splits['train']))
        print('number of val videos: ', len(self.splits['val']))
        print('number of test videos: ', len(self.splits['test']))

        for dir in opt["feats_dir"]:       
            self.feats_dir = os.path.join('/',dir)
        self.c3d_feats_dir = opt['c3d_feats_dir']
        self.with_c3d = opt['with_c3d']
        print('load feats from %s' % (self.feats_dir))
        # load in the sequence data
        self.max_len = opt["max_len"]
        print('max sequence length in data is', self.max_len)

    def __getitem__(self, ix):
        """This function returns a tuple that is further passed to collate_fn
        """
        
        if self.mode == 'train':
            filenames = os.listdir('/home/diml/hdd/feats/resnet152')
            fc_feat = []
            fc_feat.append(np.load(os.path.join('/home/diml/hdd/feats/resnet152', filenames[ix])))
            file = 'v_'+filenames[ix][:-4]
            try:
                captions = self.captions['{}'.format(file)]
                fc_feat = np.concatenate(fc_feat, axis=1)
                if self.with_c3d == 1:
                    c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy'%(ix)))
                    c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
                    fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
                label = np.zeros(self.max_len)
                label2 = np.zeros(self.max_len)
                mask = np.zeros(self.max_len)
                mask2 = np.zeros(self.max_len)
                
                gts = np.zeros((len(captions['captions'][0]), self.max_len))
                for i, cap in enumerate(captions['final_captions']):
                     if len(captions['final_captions'][cap]) > self.max_len:
                         captions['final_captions'][cap] = captions['final_captions'][cap][:self.max_len]
                         captions['final_captions'][cap][-1] = '<eos>'
                        
                     for j, w in enumerate(captions['final_captions'][cap]):
                        gts[i, j] = self.word_to_ix[w]
        
                # random select a caption for this video
                cap_ix = random.randint(0, len(captions['captions'][0]) - 1)
                cap_ix2 = 0
                while(cap_ix != cap_ix2):
                    cap_ix2 = random.randint(0, len(captions['captions'][0]) - 1)
                label = gts[cap_ix]
                label2 = gts[cap_ix2]
                non_zero = (label == 0).nonzero()
                non_zero2 = (label2 == 0).nonzero()
                mask[:int(non_zero[0][0]) + 1] = 1
                mask2[:int(non_zero2[0][0]) + 1] = 1
                data = {}
                data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
                data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
                data['labels2'] = torch.from_numpy(label2).type(torch.LongTensor)
                data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
                data['masks2'] = torch.from_numpy(mask2).type(torch.FloatTensor)
                #data['gts'] = torch.from_numpy(gts).long()
                data['video_ids'] = file
                if data is None:
                    print(data.keys())
                return data
                    
            
            except:
                print("error!")
                pass                  
        if self.mode == 'val':
            filenames = os.listdir('/home/diml/hdd/feats/resnet152_val')
            fc_feat = []
            fc_feat.append(np.load(os.path.join('/home/diml/hdd/feats/resnet152_val', filenames[ix])))
            file = 'v_'+filenames[ix][:-4]
            try:
                captions = self.captions['{}'.format(file)]
                fc_feat = np.concatenate(fc_feat, axis=1)
                if self.with_c3d == 1:
                    c3d_feat = np.load(os.path.join(self.c3d_feats_dir, 'video%i.npy'%(ix)))
                    c3d_feat = np.mean(c3d_feat, axis=0, keepdims=True)
                    fc_feat = np.concatenate((fc_feat, np.tile(c3d_feat, (fc_feat.shape[0], 1))), axis=1)
                label = np.zeros(self.max_len)
                mask = np.zeros(self.max_len)
                
                gts = np.zeros((len(captions['captions'][0]), self.max_len))
                for i, cap in enumerate(captions['final_captions']):
                     if len(captions['final_captions'][cap]) > self.max_len:
                         captions['final_captions'][cap] = captions['final_captions'][cap][:self.max_len]
                         captions['final_captions'][cap][-1] = '<eos>'
                        
                     for j, w in enumerate(captions['final_captions'][cap]):
                        gts[i, j] = self.word_to_ix[w]
        
                # random select a caption for this video
                cap_ix = random.randint(0, len(captions['captions'][0]) - 1)
                label = gts[cap_ix]
                non_zero = (label == 0).nonzero()
                mask[:int(non_zero[0][0]) + 1] = 1
                data = {}
                data['fc_feats'] = torch.from_numpy(fc_feat).type(torch.FloatTensor)
                data['labels'] = torch.from_numpy(label).type(torch.LongTensor)
                data['masks'] = torch.from_numpy(mask).type(torch.FloatTensor)
                #data['gts'] = torch.from_numpy(gts).long()
                data['video_ids'] = file
                if data is None:
                    print(data.keys())
                return data
                    
            
            except:
                print("error!")
                pass                          
    def __len__(self):
        return len(os.listdir(self.feats_dir))
