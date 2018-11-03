import pandas as pd
import matplotlib.pyplot as plt

def read_log(log_name, log_type="train"):
    log = []
    with open (log_name, "r") as f:
        for l in f:
            if log_type == "train":
                if l.startswith("Epoch"):
                    log.append(l.split(" "))
            elif log_type == "val":
                if l.startswith("Epoch"):
                    log.append(l.split(" "))

    df = pd.DataFrame(log)
    print(df.shape)
    return df

if __name__ == '__main__':
    log = []
    simpleRNN = read_log("../log/log_lab_machine/simpleRNN.txt")
    bag_of_words = read_log("../log/log_lab_machine/bag_of_words.txt")
    gru_summarunner = read_log("../log/log_lab_machine/gru_summarunner.txt")
    lstm_summarunner = read_log("../log/log_lab_machine/lstm_summarunner.txt")
    # train1 = read_log("../log/train1_lf.txt")
    # log_name = "../log/simpleRNN.txt"
    # with open (log_name, "r") as f:
    #     data = f.readlines()
    #     print(data)
    #     l = f.readline()
    #     print(l)
    #     if l.startswith("Epoch"):
    #         log.append(l)
    #     print(log)
    col_num = 5
    # print(bag_of_words[col_num])
    plt.plot(bag_of_words[col_num])
    plt.ylabel('average of rouge 1,2,L f-scores')
    # plt.show(block=True)
    # simpleRNN[col_num],gru_summarunner[col_num],lstm_summarunner[col_num]