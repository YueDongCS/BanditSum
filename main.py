#!/usr/bin/env python
# coding:utf8

from __future__ import print_function

import argparse
import logging

import torch
from torch.autograd import Variable

import evaluate
import model
from dataLoader import *
from helper import Config, tokens_to_sentences, prepare_data
from reinforce import ReinforceReward

np.set_printoptions(precision=4, suppress=True)


# ../model/summary.model.simpleRNN.avg_f.False.batch_avg.oracle_l.3.
# bsz.20.rl_loss.2.train_example_quota.-1.length_limit.-1.data.CNN_DM_pickle_data.
def extractive_training(args, vocab):
    print(args)
    print("generating config")
    config = Config(
        vocab_size=vocab.embedding.shape[0],
        embedding_dim=vocab.embedding.shape[1],
        position_size=500,
        position_dim=50,
        word_input_size=100,
        sent_input_size=2 * args.hidden,
        word_GRU_hidden_units=args.hidden,
        sent_GRU_hidden_units=args.hidden,
        pretrained_embedding=vocab.embedding,
        word2id=vocab.w2i,
        id2word=vocab.i2w,
        dropout=args.dropout,
    )
    model_name = ".".join((args.model_file,
                           str(args.ext_model),
                           str(args.rouge_metric), str(args.std_rouge),
                           str(args.rl_baseline_method), "oracle_l", str(args.oracle_length),
                           "bsz", str(args.batch_size), "rl_loss", str(args.rl_loss_method),
                           "train_example_quota", str(args.train_example_quota),
                           "length_limit", str(args.length_limit),
                           "data", os.path.split(args.data_dir)[-1],
                           "hidden", str(args.hidden),
                           "dropout", str(args.dropout),
                           'ext'))
    print(model_name)

    log_name = ".".join(("../log/model",
                         str(args.ext_model),
                         str(args.rouge_metric), str(args.std_rouge),
                         str(args.rl_baseline_method), "oracle_l", str(args.oracle_length),
                         "bsz", str(args.batch_size), "rl_loss", str(args.rl_loss_method),
                         "train_example_quota", str(args.train_example_quota),
                         "length_limit", str(args.length_limit),
                         "hidden", str(args.hidden),
                         "dropout", str(args.dropout),
                         'ext'))

    print("init data loader and RL learner")
    data_loader = PickleReader(args.data_dir)

    # init statistics
    reward_list = []
    best_eval_reward = 0.
    model_save_name = model_name

    if args.fine_tune:
        model_save_name = model_name + ".fine_tune"
        log_name = log_name + ".fine_tune"
        args.std_rouge = True
        print("fine_tune model with std_rouge, args.std_rouge changed to %s" % args.std_rouge)

    reinforce = ReinforceReward(std_rouge=args.std_rouge, rouge_metric=args.rouge_metric,
                                b=args.batch_size, rl_baseline_method=args.rl_baseline_method,
                                loss_method=1)

    print('init extractive model')

    if args.ext_model == "lstm_summarunner":
        extract_net = model.SummaRuNNer(config)
    elif args.ext_model == "gru_summarunner":
        extract_net = model.GruRuNNer(config)
    elif args.ext_model == "bag_of_words":
        extract_net = model.SimpleRuNNer(config)
    elif args.ext_model == "simpleRNN":
        extract_net = model.SimpleRNN(config)
    elif args.ext_model == "RNES":
        extract_net = model.RNES(config)
    elif args.ext_model == "Refresh":
        extract_net = model.Refresh(config)
    elif args.ext_model == "simpleCONV":
        extract_net = model.simpleCONV(config)
    else:
        print("this is no model to load")

    extract_net.cuda()

    # print("current model name: %s"%model_name)
    # print("current log file: %s"%log_name)

    logging.basicConfig(filename='%s.log' % log_name,
                        level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
    if args.load_ext:
        print("loading existing model%s" % model_name)
        extract_net = torch.load(model_name, map_location=lambda storage, loc: storage)
        extract_net.cuda()
        print("finish loading and evaluate model %s" % model_name)
        # evaluate.ext_model_eval(extract_net, vocab, args, eval_data="test")
        best_eval_reward, _ = evaluate.ext_model_eval(extract_net, vocab, args, "val")

    # Loss and Optimizer
    optimizer_ext = torch.optim.Adam(extract_net.parameters(), lr=args.lr, betas=(0., 0.999))

    print("starting training")
    n_step = 100
    for epoch in range(args.epochs_ext):
        train_iter = data_loader.chunked_data_reader("train", data_quota=args.train_example_quota)
        step_in_epoch = 0
        for dataset in train_iter:
            for step, docs in enumerate(BatchDataLoader(dataset, shuffle=True)):
                try:
                    extract_net.train()
                    # if True:
                    step_in_epoch += 1
                    # for i in range(1):  # how many times a single data gets updated before proceeding
                    doc = docs[0]
                    doc.content = tokens_to_sentences(doc.content)
                    doc.summary = tokens_to_sentences(doc.summary)
                    if args.oracle_length == -1:  # use true oracle length
                        oracle_summary_sent_num = len(doc.summary)
                    else:
                        oracle_summary_sent_num = args.oracle_length

                    x = prepare_data(doc, vocab)
                    if min(x.shape) == 0:
                        continue
                    sents = Variable(torch.from_numpy(x)).cuda()
                    outputs = extract_net(sents)

                    if args.prt_inf and np.random.randint(0, 100) == 0:
                        prt = True
                    else:
                        prt = False

                    loss, reward = reinforce.train(outputs, doc,
                                                   max_num_of_sents=oracle_summary_sent_num,
                                                   max_num_of_bytes=args.length_limit,
                                                   prt=prt)
                    if prt:
                        print('Probabilities: ', outputs.squeeze().data.cpu().numpy())
                        print('-' * 80)

                    reward_list.append(reward)

                    if isinstance(loss, Variable):
                        loss.backward()

                    if step % 1 == 0:
                        torch.nn.utils.clip_grad_norm(extract_net.parameters(), 1)  # gradient clipping
                        optimizer_ext.step()
                        optimizer_ext.zero_grad()
                    # print('Epoch %d Step %d Reward %.4f'%(epoch,step_in_epoch,reward))
                    logging.info('Epoch %d Step %d Reward %.4f' % (epoch, step_in_epoch, reward))
                except Exception as e:
                    print(e)

                if (step_in_epoch) % n_step == 0 and step_in_epoch != 0:
                    print('Epoch ' + str(epoch) + ' Step ' + str(step_in_epoch) +
                          ' reward: ' + str(np.mean(reward_list)))
                    reward_list = []

                if (step_in_epoch) % 10000 == 0 and step_in_epoch != 0:
                    print("doing evaluation")
                    extract_net.eval()
                    eval_reward, lead3_reward = evaluate.ext_model_eval(extract_net, vocab, args, "val")
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        print("saving model %s with eval_reward:" % model_save_name, eval_reward, "leadreward",
                              lead3_reward)
                        torch.save(extract_net, model_name)
                    print('epoch ' + str(epoch) + ' reward in validation: '
                          + str(eval_reward) + ' lead3: ' + str(lead3_reward))
    return extract_net


def main():
    torch.manual_seed(233)
    parser = argparse.ArgumentParser()

    parser.add_argument('--vocab_file', type=str, default='../data/CNN_DM_pickle_data/vocab_100d.p')
    parser.add_argument('--data_dir', type=str, default='../data/CNN_DM_pickle_data')
    parser.add_argument('--model_file', type=str, default='../model/summary.model')
    parser.add_argument('--epochs_ext', type=int, default=10)
    parser.add_argument('--load_ext', action='store_true')
    parser.add_argument('--hidden', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--device', type=int, default=0,
                        help='select GPU')
    parser.add_argument('--std_rouge', action='store_true')

    parser.add_argument('--oracle_length', type=int, default=3,
                        help='-1 for giving actual oracle number of sentences'
                             'otherwise choose a fixed number of sentences')
    parser.add_argument('--rouge_metric', type=str, default='avg_f')
    parser.add_argument('--rl_baseline_method', type=str, default="batch_avg",
                        help='greedy, global_avg, batch_avg, batch_med, or none')
    parser.add_argument('--rl_loss_method', type=int, default=2,
                        help='1 for computing 1-log on positive advantages,'
                             '0 for not computing 1-log on all advantages')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--fine_tune', action='store_true', help='fine tune with std rouge')
    parser.add_argument('--train_example_quota', type=int, default=-1,
                        help='how many train example to train on: -1 means full train data')
    parser.add_argument('--length_limit', type=int, default=-1,
                        help='length limit output')
    parser.add_argument('--ext_model', type=str, default="simpleRNN",
                        help='lstm_summarunner, gru_summarunner, bag_of_words, simpleRNN')
    parser.add_argument('--prt_inf', action='store_true')

    args = parser.parse_args()

    if args.length_limit > 0:
        args.oracle_length = 2

    torch.cuda.set_device(args.device)

    print('generate config')
    with open(args.vocab_file, "rb") as f:
        vocab = pickle.load(f)
    print(vocab)

    extract_net = extractive_training(args, vocab)


if __name__ == '__main__':
    main()
