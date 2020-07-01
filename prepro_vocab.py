import re
import json
import argparse
import numpy as np
import os

def build_vocab(vids, params):
    count_thr = params['word_count_threshold']
    # count up the number of words
    counts = {}
    for vid, caps in vids.items():
        for cap in caps['captions']:
            cap = str(cap[:-1]).lower()
            ws = re.sub(r'[.!,;?"]', ' ', cap).split()
            for w in ws:
                counts[w] = counts.get(w, 0) + 1
    # cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
# =============================================================================
#     vocab = [w for w, n in counts.items()]
# =============================================================================
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')
    for vid, caps in vids.items():
        
        caps = caps['captions']
        vids[vid]['final_captions'] = {}
        for i, cap in enumerate(caps[0]):
            vids[vid]['final_captions']['{}'.format(i)] = []
            cap = str(cap[:-1]).lower()
            ws = re.sub(r'[.!,;?"[]]', ' ', cap).split()
# =============================================================================
#             caption = [
#                 '<sos>'] + [w for w in ws] + ['<eos>']
# =============================================================================
            caption = [
                '<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
            vids[vid]['final_captions']['{}'.format(i)] = caption
    return vocab,vids


def main(params):
    videos = json.load(open(params['input_json'], 'r'))['sentence']
    video_caption = {}
    for k,i in enumerate(videos):
        # if i['video_id'] == 'v_vt81bZ6_GcQ':
        #     print("check")
        video_caption[i['video_id']] = {'captions': []}
        video_caption[i['video_id']]['captions'].append(i['captions'])

    # create the vocab
    vocab, video_caption = build_vocab(video_caption, params)
    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'
    

 
    out = {}
    out['ix_to_word'] = itow
    out['word_to_ix'] = wtoi
    out['videos'] = {'train': [], 'val': [], 'test': []}
    videoss = json.load(open(params['input_json'], 'r'))['videos']
    for i in videoss:
        out['videos'][i['split']].append(i['video_id'])
    with open('info1.json', 'w', encoding="utf-8") as make_file:
        json.dump(out, make_file, ensure_ascii=False, indent="\t")
    with open('caption1.json', 'w', encoding="utf-8") as make_file:
        json.dump(video_caption, make_file, ensure_ascii=False, indent="\t")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', type=str, default='./input.json',
                        help='msr_vtt videoinfo json')
    parser.add_argument('--info_json', default='data/info.json',help='info about iw2word and word2ix')
    parser.add_argument('--caption_json', default='data/caption.json', help='caption json file')


    parser.add_argument('--word_count_threshold', default=1, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
