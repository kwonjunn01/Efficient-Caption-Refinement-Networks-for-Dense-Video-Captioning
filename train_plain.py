import json
import os
import matplotlib.pylab as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import misc.utils as utils
import opts
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from dataloader import VideoDataset
from misc.rewards import get_self_critical_reward, init_cider_scorer
from models import DecoderRNN, EncoderRNN, S2VTAttModel, S2VTModel, nEncoderRNN
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange, tqdm


def train(dataset, loader, model, crit, optimizer, lr_scheduler, opt, rl_crit=None):
    writer = SummaryWriter('./runs/video_caption_basic')
    model.load_state_dict(torch.load('/home/diml/video-caption.pytorch/save/new_model_200.pth'))
    #model = nn.DataParallel(model)
    model.train()
    vocab = dataset.get_vocab()

    for epoch in trange(300):
        t_loss = 0
# =============================================================================
#         model.eval()
#         ev.demov(model,crit, dataset, dataset.get_vocab(),opt)
# =============================================================================
        
        lr_scheduler.step()
        iteration = 0
        
        # If start self crit training
        if opt["self_crit_after"] != -1 and epoch >= opt["self_crit_after"]:
            sc_flag = True
            init_cider_scorer(opt["cached_tokens"])
        else:
            sc_flag = False

        for idx ,data in enumerate(loader):
            torch.cuda.synchronize()
            fc_feats = data['fc_feats'].cuda()
            labels = data['labels'].cuda()
            masks = data['masks'].cuda()
            optimizer.zero_grad()
            if not sc_flag:
                seq_probs, seq_preds, hn,de_hn= model(fc_feats, labels, 'train')
                loss_C = crit(seq_probs, labels[:, 1:], masks[:, 1:])
                
                loss = loss_C
            else:
                seq_probs, seq_preds = model(
                    fc_feats, mode='inference', opt=opt)
                reward = get_self_critical_reward(model, fc_feats, data,
                                                  seq_preds)
                print(reward.shape)
                loss = rl_crit(seq_probs, seq_preds,
                               torch.from_numpy(reward).float().cuda())
                
            t_loss += loss.item()
            loss.backward()
            clip_grad_value_(model.parameters(), opt['grad_clip'])
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            iteration += 1
            if not sc_flag:
                print("iter %d (epoch %d), train_loss = %.6f" %
                      (iteration, epoch, train_loss))
            else:
                print("iter %d (epoch %d), avg_reward = %.6f" %
                      (iteration, epoch+201, np.mean(reward[:, 0])))
        writer.add_scalar('training total loss',
                        t_loss / 140,
                        epoch+200)
        if epoch % opt["save_checkpoint_every"] == 0:
            
            model_path = os.path.join(opt["checkpoint_path"],
                                      'new_model_%d.pth' % (epoch+ 200))
            model_info_path = os.path.join(opt["checkpoint_path"],
                                           'Rnew_model_score.txt')
            torch.save(model.state_dict(), model_path)
            print("model saved to %s" % (model_path))
            
            with open(model_info_path, 'a') as f:
                f.write("model_%d, loss: %.6f\n" % (epoch, train_loss))
                
        with torch.no_grad():
           _, seq_preds,__,___ = model(fc_feats, mode='inference', opt=opt)
           print(utils.decode_sequence(vocab, seq_preds)[0])



def main(opt):
    
    dataset = VideoDataset(opt, 'train')
    dataloader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=True)
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            opt['dim_vid'],
            rnn_cell=opt['rnn_type'],
            n_layers=opt['num_layers'],
            rnn_dropout_p=opt["rnn_dropout_p"])
    elif opt["model"] == "S2VTAttModel":
        encoder_A = EncoderRNN(
            opt["dim_vid"],
            opt["dim_hidden"],
            bidirectional=bool(opt["bidirectional"]),
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"])
        decoder_A = DecoderRNN(
            opt["vocab_size"],
            opt["max_len"],
            opt["dim_hidden"],
            opt["dim_word"],
            input_dropout_p=opt["input_dropout_p"],
            rnn_cell=opt['rnn_type'],
            rnn_dropout_p=opt["rnn_dropout_p"],
            bidirectional=bool(opt["bidirectional"]))
        model = S2VTAttModel(encoder_A, decoder_A)
    
    model = model.cuda()
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    params = list(model.parameters())
    optimizer = optim.Adam(
        params,
        lr=opt["learning_rate"],
        weight_decay=opt["weight_decay"])
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opt["learning_rate_decay_every"],
        gamma=opt["learning_rate_decay_rate"])

    train(dataset, dataloader, model, crit, optimizer, exp_lr_scheduler, opt, rl_crit)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    opt_json = os.path.join(opt["checkpoint_path"], 'opt_info.json')
    if not os.path.isdir(opt["checkpoint_path"]):
        os.mkdir(opt["checkpoint_path"])
    with open(opt_json, 'w') as f:
        json.dump(opt, f)
    print('save opt details to %s' % (opt_json))
    main(opt)
