from __future__ import print_function

import argparse
import time

import torch
import numpy as np

import dataLoader
import helper
from helper import tokens_to_sentences
from reinforce import return_summary_index
from rougefonc import from_summary_index_generate_hyp_ref, RougeTest_pyrouge, RougeTest_rouge

np.set_printoptions(precision=4, suppress=True)


def reinforce_loss(probs, doc, id=0,
                   max_num_of_sents=3, max_num_of_bytes=-1,
                   std_rouge=False, rouge_metric="all", compute_score=True):
    # sample sentences
    probs_numpy = probs.data.cpu().numpy()
    probs_numpy = np.reshape(probs_numpy, len(probs_numpy))
    max_num_of_sents = min(len(probs_numpy), max_num_of_sents)  # max of sents# in doc and sents# in summary

    rl_baseline_summary_index, _ = return_summary_index(probs_numpy, probs,
                                                        sample_method="greedy", max_num_of_sents=max_num_of_sents)
    rl_baseline_summary_index = sorted(rl_baseline_summary_index)
    rl_baseline_hyp, rl_baseline_ref = from_summary_index_generate_hyp_ref(doc, rl_baseline_summary_index)

    lead3_hyp, lead3_ref = from_summary_index_generate_hyp_ref(doc, range(max_num_of_sents))

    if std_rouge:
        rl_baseline_reward = RougeTest_pyrouge(rl_baseline_ref, rl_baseline_hyp, id=id, rouge_metric=rouge_metric,
                                               compute_score=compute_score, path=os.path.join('./result/rl'),
                                               max_num_of_bytes=max_num_of_bytes)
        lead3_reward = RougeTest_pyrouge(lead3_ref, lead3_hyp, id=id, rouge_metric=rouge_metric,
                                         compute_score=compute_score, path=os.path.join('./result/lead'),
                                         max_num_of_bytes=max_num_of_bytes)
    else:
        rl_baseline_reward = RougeTest_rouge(rl_baseline_ref, rl_baseline_hyp, rouge_metric,
                                             max_num_of_bytes=max_num_of_bytes)
        lead3_reward = RougeTest_rouge(lead3_ref, lead3_hyp, rouge_metric, max_num_of_bytes=max_num_of_bytes)

    return rl_baseline_reward, lead3_reward


def ext_model_eval(model, vocab, args, eval_data="test"):
    print("loading data %s" % eval_data)

    model.eval()

    data_loader = dataLoader.PickleReader(args.data_dir)
    eval_rewards, lead3_rewards = [], []
    data_iter = data_loader.chunked_data_reader(eval_data)
    print("doing model evaluation on %s" % eval_data)

    for phase, dataset in enumerate(data_iter):
        for step, docs in enumerate(dataLoader.BatchDataLoader(dataset, shuffle=False)):
            print("Done %2d chunck, %4d/%4d doc\r" % (phase + 1, step + 1, len(dataset)), end='')

            doc = docs[0]
            doc.content = tokens_to_sentences(doc.content)
            doc.summary = tokens_to_sentences(doc.summary)
            if len(doc.content) == 0 or len(doc.summary) == 0:
                continue

            # if doc.content[0].find('CNN') >= 0:
            #     args.oracle_length = 3
            # else:
            #     args.oracle_length = 4

            if args.oracle_length == -1:  # use true oracle length
                oracle_summary_sent_num = len(doc.summary)
            else:
                oracle_summary_sent_num = args.oracle_length

            x = helper.prepare_data(doc, vocab)
            if min(x.shape) == 0:
                continue
            sents = torch.autograd.Variable(torch.from_numpy(x)).cuda()

            outputs = model(sents)

            compute_score = (step == len(dataset) - 1) or (args.std_rouge is False)
            if eval_data == "test":
                # try:
                reward, lead3_r = reinforce_loss(outputs, doc, id=phase * 1000 + step,
                                                 max_num_of_sents=oracle_summary_sent_num,
                                                 max_num_of_bytes=args.length_limit,
                                                 std_rouge=args.std_rouge, rouge_metric="all",
                                                 compute_score=compute_score)
            else:
                reward, lead3_r = reinforce_loss(outputs, doc, id=phase * 1000 + step,
                                                 max_num_of_sents=oracle_summary_sent_num,
                                                 max_num_of_bytes=args.length_limit,
                                                 std_rouge=args.std_rouge, rouge_metric=args.rouge_metric,
                                                 compute_score=compute_score)

            if compute_score:
                eval_rewards.append(reward)
                lead3_rewards.append(lead3_r)

    avg_eval_r = np.mean(eval_rewards, axis=0)
    avg_lead3_r = np.mean(lead3_rewards, axis=0)
    print('model %s reward in %s:' % (args.rouge_metric, eval_data))
    print(avg_eval_r)
    print(avg_lead3_r)
    return avg_eval_r, avg_lead3_r


if __name__ == '__main__':
    from dataLoader import *

    torch.manual_seed(233)
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_file', type=str, default='../data/CNN_DM_pickle_data/vocab_100d.p')
    parser.add_argument('--data_dir', type=str, default='../data/CNN_DM_pickle_data/')
    parser.add_argument('--model_file', type=str, default='../model/summary.ext')

    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--std_rouge', action='store_true')

    parser.add_argument('--oracle_length', type=int, default=3,
                        help='-1 for giving actual oracle number of sentences'
                             'otherwise choose a fixed number of sentences')
    parser.add_argument('--rouge_metric', type=str, default='all')
    parser.add_argument('--rl_baseline_method', type=str, default="greedy",
                        help='greedy, global_avg,batch_avg,or none')
    parser.add_argument('--length_limit', type=int, default=-1,
                        help='length limit output')

    args = parser.parse_args()

    torch.cuda.set_device(args.device)

    print('generate config')
    with open(args.vocab_file, "rb") as f:
        vocab = pickle.load(f)
    print(vocab)

    print("loading existing model%s" % args.model_file)
    extract_net = torch.load(args.model_file, map_location=lambda storage, loc: storage)
    extract_net.cuda()
    print("finish loading and evaluate model %s" % args.model_file)

    start_time = time.time()
    ext_model_eval(extract_net, vocab, args, eval_data="test")
    print('Test time:', time.time() - start_time)
