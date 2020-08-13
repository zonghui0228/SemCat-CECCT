import os
import six
import collections



# 创建一个数据字典。数字字典就是把每个字都对应一个一个数字，包括标点符号。
def build_dict(train_data_path, test_data_path, dict_path, cutoff=0):
    word_freq = collections.defaultdict(int)
    # 读取已经训练数据
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = [line.strip().split("\t")[2] for line in f]
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = [line.strip().split("\t")[2] for line in f]

    for data in train_data+test_data:
        for word in data:
            word_freq[word] += 1
    # Not sure if we should prune less-frequent words here.
    word_freq = [x for x in six.iteritems(word_freq) if x[1] > cutoff]
    dictionary = sorted(word_freq, key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*dictionary))
    word_idx = dict(list(zip(words, six.moves.range(len(words)))))
    word_idx['<unk>'] = len(words)

    # 把这些字典保存到本地中
    with open(dict_path, 'w', encoding='utf-8') as f:
        f.write(str(word_idx))
    print("数据字典生成完成！")

# 获取字典的长度
def get_dict_len(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        line = eval(f.readlines()[0])
    return len(line.keys())

def create_data_list(data_root_path):
    with open(os.path.join(data_root_path, 'dict.txt'), 'r', encoding='utf-8') as f_dict:
        dict_txt = eval(f_dict.readlines()[0])
    print("字典长度：{}".format(len(dict_txt)))
    with open(os.path.join(data_root_path, 'tags.txt'), 'r', encoding='utf-8') as f_tag:
        tag_txt = eval(f_tag.readlines()[0])
    print("类别数目：{}".format(len(tag_txt)))
    with open(os.path.join(data_root_path, 'train_data.txt'), 'r', encoding='utf-8') as f_train:
        lines = f_train.readlines()
    with open(os.path.join(data_root_path, 'train_list.txt'), 'w', encoding='utf-8') as f_train_list:
        for line in lines:
            labs = ""
            l = line.strip().split("\t")
            for word in l[2]:
                labs += str(dict_txt[word]) + ','
            f_train_list.write(labs + '\t' + str(tag_txt[l[1]]) + '\n')

    with open(os.path.join(data_root_path, 'test_data.txt'), 'r', encoding='utf-8') as f_test:
        lines = f_test.readlines()
    with open(os.path.join(data_root_path, 'test_list.txt'), 'w', encoding='utf-8') as f_test_list:
        for line in lines:
            labs = ""
            l = line.strip().split("\t")
            for word in l[2]:
                labs += str(dict_txt[word]) + ','
            f_test_list.write(labs + '\t' + str(tag_txt[l[1]]) + '\n')
    print("数据列表生成完成！")

if __name__ == "__main__":
    data_root_path = "./data/"
    train_data_path = "./data/train_data.txt"
    test_data_path = "./data/test_data.txt"
    dict_path = "./data/dict.txt"
    # build_dict(train_data_path, test_data_path, dict_path)
    # print(get_dict_len(dict_path))
    # create_data_list(data_root_path)
