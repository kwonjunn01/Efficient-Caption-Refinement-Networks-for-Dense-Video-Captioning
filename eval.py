import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel,nEncoderRNN
from dataloader import VideoDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer

from pandas.io.json import json_normalize


def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["captions"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'captions': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'captions': row[0]})
    return gts

def demov(model, crit, dataset, vocab, opt):
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=False)
    for i,data in enumerate(loader):
        # forward the model to get loss
        fc_feats = data['fc_feats'].cuda()
      
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds = model(
                fc_feats, mode='inference', opt=opt)

        sents = utils.decode_sequence(vocab, seq_preds)
        
        print(sents)
        if i == 0:
            break
def test(model,rem, crit, dataset, vocab, opt):
    videos = json.load(open('caption1.json', 'r'))
    model.eval()
    loader = DataLoader(dataset, batch_size=opt["batch_size"], shuffle=False)
    scorer = COCOScorer()
    gt_dataframe = json_normalize(
        json.load(open(opt["input_json"]))['sentence'])
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    results = []
    results_f = []
    results_avg = []
    samples = {}
    samples_f = {}
    samples_avg = {}
    sample_all = {}
    for i,data in enumerate(loader):
        # forward the model to get loss
        fc_feats = data['fc_feats'].cuda()
        labels = data['labels'].cuda()
        masks = data['masks'].cuda()
        video_ids = data['video_ids']
        
      
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds, en_hn, de_hn = model(
                fc_feats, mode='inference', opt=opt)
            fake_en_hn= rem(de_hn, seq_probs)
            f_seq_probs, f_seq_preds, __, __ = model(
                fc_feats, mode='inference',h=fake_en_hn, opt=opt)
            avg_en_hn = (en_hn+fake_en_hn)/2
            avg_f_seq_probs,avg_f_seq_preds, __, __ = model(
                fc_feats, mode='inference',h=avg_en_hn, opt=opt)
        sents = utils.decode_sequence(vocab, seq_preds)
        f_sents = utils.decode_sequence(vocab, f_seq_preds)
        avg_sents = utils.decode_sequence(vocab, avg_f_seq_preds)

        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'captions': sent}]
            samples_f[video_id] = [{'image_id': video_id, 'captions': f_sents[k]}]
            samples_avg[video_id] = [{'image_id': video_id, 'captions': avg_sents[k]}]
            sample_all[video_id] = [{'ground truth': videos[video_id]['captions'], 'caption_origin':sent, 'caption_fake':f_sents[k], 'caption_average':avg_sents[k]}]
            
        if i>1: 
            print(seq_preds.size())
            break

    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
        valid_score_f = scorer.score(gts, samples_f, samples_f.keys())
        valid_score_avg = scorer.score(gts, samples_avg, samples_avg.keys())
    results.append(valid_score)
    results_f.append(valid_score_f)
    results_avg.append(valid_score_avg)
    print(valid_score)

    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])
    
    with open(os.path.join(opt["results_path"], "scores2.txt"), 'w') as scores_table:
        scores_table.write(json.dumps(results[0]) + "\n")
    with open(os.path.join(opt["results_path"], "scores_f2.txt"), 'w') as scores_table:
        scores_table.write(json.dumps(results_f[0]) + "\n")
    with open(os.path.join(opt["results_path"], "scores_avg2.txt"), 'w') as scores_table:
        scores_table.write(json.dumps(results_avg[0]) + "\n")
    with open(os.path.join(opt["results_path"],
                           opt["model"].split("/")[-1].split('.')[0] + "2.json"), 'w') as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score},
                  prediction_results)
    with open(os.path.join(opt["results_path"],
                           opt["model"].split("/")[-1].split('.')[0] + "_f2.json"), 'w') as prediction_results:
        json.dump({"predictions": samples_f, "scores": valid_score_f},
                  prediction_results)
    with open(os.path.join(opt["results_path"],
                           opt["model"].split("/")[-1].split('.')[0] + "_avg2.json"), 'w') as prediction_results:
        json.dump({"predictions": samples_avg, "scores": valid_score_avg},
                  prediction_results)
    with open('./results/total_caption2.json', 'w') as f:
        json.dump({"total": sample_all},f)



def main(opt):
    dataset = VideoDataset(opt, "val")
    opt["vocab_size"] = dataset.get_vocab_size()
    opt["seq_length"] = dataset.max_len
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                          rnn_dropout_p=opt["rnn_dropout_p"]).cuda()
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"], bidirectional=opt["bidirectional"],
                             input_dropout_p=opt["input_dropout_p"], rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(opt["vocab_size"], opt["max_len"], opt["dim_hidden"], opt["dim_word"],
                             input_dropout_p=opt["input_dropout_p"],
                             rnn_dropout_p=opt["rnn_dropout_p"], bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder).cuda()
    #model = nn.DataParallel(model)
    # Setup the model
    rem = nEncoderRNN.nEncoderRNN()
    rem.load_state_dict(torch.load("./save/RECON222_module_200.pth"))
    rem= rem.cuda()
    model.load_state_dict(torch.load(opt["saved_model"]))
    crit = utils.LanguageModelCriterion()
    test(model,rem, crit, dataset, dataset.get_vocab(), opt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recover_opt', type=str, default='./save/opt_info.json',
                        help='recover train opts from saved opt_json')
    parser.add_argument('--saved_model', type=str, default='./save/RECON222_model_200.pth',
                        help='path to saved model to evaluate')

    parser.add_argument('--dump_json', type=int, default=1,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--results_path', type=str, default='./results/')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu device number')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='minibatch size')
    parser.add_argument('--sample_max', type=int, default=1,
                        help='0/1. whether sample max probs  to get next word in inference stage')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1. Usually 2 or 3 works well.')

    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    opt['bidirectional'] = bool(0)
    for k, v in args.items():
        opt[k] = v
    os.environ['CUDA_VISIBLE_DEVICES'] = opt["gpu"]
    main(opt)
