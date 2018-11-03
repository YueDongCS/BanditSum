import shutil
import os
import codecs

from pyrouge import Rouge155
from rouge import Rouge

rouge = Rouge()


def RougeTest_rouge(ref, hyp, rouge_metric="all", max_num_of_bytes=-1):
    ref = [_.decode('ascii', 'ignore').encode('utf-8').lower() for _ in ref]
    hyp = [_.decode('ascii', 'ignore').encode('utf-8').lower() for _ in hyp]

    if max_num_of_bytes > 0:
        ref = cutwords(ref)
        hyp = cutwords(hyp)

    rouge_score = rouge.get_scores([hyp], [ref])
    if rouge_metric[1] == 'f':
        return rouge_score[0]['rouge-%s' % rouge_metric[0]]['f']
    elif rouge_metric[1] == 'r':
        return rouge_score[0]['rouge-%s' % rouge_metric[0]]['r']
    elif rouge_metric == 'avg_f':
        return (rouge_score[0]['rouge-1']['f'] + rouge_score[0]['rouge-2']['f'] + rouge_score[0]['rouge-l']['f']) / 3
    elif rouge_metric == 'avg_r':
        return (rouge_score[0]['rouge-1']['r'] + rouge_score[0]['rouge-2']['r'] + rouge_score[0]['rouge-l']['r']) / 3
    else:
        return (rouge_score[0]['rouge-1']['p'], rouge_score[0]['rouge-1']['r'], rouge_score[0]['rouge-1']['f'],
                rouge_score[0]['rouge-2']['p'], rouge_score[0]['rouge-2']['r'], rouge_score[0]['rouge-2']['f'],
                rouge_score[0]['rouge-l']['p'], rouge_score[0]['rouge-l']['r'], rouge_score[0]['rouge-l']['f'])


home_path = os.path.expanduser('~')

def RougeTest_pyrouge(ref, hyp, id=0, rouge_metric='all', compute_score=True,
                      path='./result', max_num_of_bytes=-1):
    # initialization
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists(path):
        os.mkdir(path)

    # write new ref and hyp
    with codecs.open(os.path.join(path, 'ref.' + str(id) + '.txt'), 'w', encoding="UTF-8") as f:
        f.write(Rouge155.convert_text_to_rouge_format('\n'.join(ref).decode('UTF-8', 'ignore')))
    with codecs.open(os.path.join(path, 'hyp.' + str(id) + '.txt'), 'w', encoding="UTF-8") as f:
        f.write(Rouge155.convert_text_to_rouge_format('\n'.join(hyp).decode('UTF-8', 'ignore')))

    if compute_score:
        if max_num_of_bytes > 0:
            r = Rouge155('%s/SciSoft/ROUGE-1.5.5/' % home_path,
                         '-e %s/SciSoft/ROUGE-1.5.5/data -a -c 95 -m -n 2 -b %d' % (home_path, max_num_of_bytes))
        else:
            r = Rouge155('%s/SciSoft/ROUGE-1.5.5/' % home_path,
                         '-e %s/SciSoft/ROUGE-1.5.5/data -a -c 95 -m -n 2' % home_path)
        r.system_dir = path
        r.model_dir = path
        r.system_filename_pattern = 'hyp.(\d+).txt'
        r.model_filename_pattern = 'ref.#ID#.txt'

        output = r.evaluate()
        # print(output)
        output_dict = r.output_to_dict(output)
        # cleanup
        shutil.rmtree(path)
        shutil.rmtree(r._config_dir)

        if rouge_metric[1] == 'f':
            return output_dict["rouge_%s_f_score" % rouge_metric[0]]
        elif rouge_metric[1] == 'r':
            return output_dict["rouge_%s_recall" % rouge_metric[0]]
        elif rouge_metric == 'avg_f':
            return (output_dict["rouge_1_f_score"] + output_dict["rouge_2_f_score"] + output_dict[
                "rouge_l_f_score"]) / 3
        elif rouge_metric == 'avg_r':
            return (output_dict["rouge_1_recall"] + output_dict["rouge_2_recall"] + output_dict["rouge_l_recall"]) / 3
        else:
            return (output_dict["rouge_1_precision"], output_dict["rouge_1_recall"], output_dict["rouge_1_f_score"],
                    output_dict["rouge_2_precision"], output_dict["rouge_2_recall"], output_dict["rouge_2_f_score"],
                    output_dict["rouge_l_precision"], output_dict["rouge_l_recall"], output_dict["rouge_l_f_score"])
    else:
        return None


def cutwords(sens, max_num_of_chars):
    output = []
    quota = max_num_of_chars
    for sen in sens:
        if quota > len(sen):
            output.append(sen)
            quota -= len(sen)
        else:
            output.append(sen[:quota])
            break
    return output


def from_summary_index_generate_hyp_ref(doc, summary_index):
    hyp = [doc.content[i].strip() for i in summary_index]
    ref = [s.strip() for s in doc.summary]

    return hyp, ref


def from_summary_index_compute_rouge(doc, summary_index, std_rouge=False, rouge_metric="all", max_num_of_bytes=-1):
    # greedy approach directly use this

    hyp, ref = from_summary_index_generate_hyp_ref(doc, summary_index)

    if len(hyp) == 0 or len(ref) == 0:
        return 0.

    if std_rouge:
        score = RougeTest_pyrouge(ref, hyp, rouge_metric=rouge_metric, max_num_of_bytes=max_num_of_bytes)
    else:
        score = RougeTest_rouge(ref, hyp, rouge_metric=rouge_metric, max_num_of_bytes=max_num_of_bytes)
    return score
