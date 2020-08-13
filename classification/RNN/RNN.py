# 使用文本卷积网络模型，分类中文临床试验筛选标准Criteria
# 类别有44类
# 包括模型训练，模型保存，测试集评估


# 引入要用到的库和定义全局变量
from __future__ import print_function
import os
import sys
import math
import codecs
import numpy as np

import paddle
import paddle.fluid as fluid



#栈式双向LSTM
def stacked_lstm_net(data, input_dim, class_dim, emb_dim, hid_dim, stacked_num):

    #计算词向量
    emb = fluid.embedding(
        input=data, size=[input_dim, emb_dim], is_sparse=True)

    #第一层栈
    #全连接层
    fc1 = fluid.layers.fc(input=emb, size=hid_dim)
    #lstm层
    lstm1, cell1 = fluid.layers.dynamic_lstm(input=fc1, size=hid_dim)

    inputs = [fc1, lstm1]

    #其余的所有栈结构
    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hid_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hid_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    #池化层
    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    #全连接层，softmax预测
    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last], size=class_dim, act='softmax')
    return prediction


# 定义预测程序（inference_program）。预测程序使用convolution_net来对fluid.layer.data的输入进行预测
def inference_program(word_dict):
    data = fluid.data(name="words", shape=[None], dtype="int64", lod_level=1)
    dict_dim = len(word_dict)
    net = stacked_lstm_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM, STACKED_NUM)
    return net


# 定义了training_program。它使用了从inference_program返回的结果来计算误差。。
def train_program(prediction):
    label = fluid.data(name="label", shape=[None, 1], dtype="int64")
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)
    return [avg_cost, accuracy]   #返回平均cost和准确率acc

# 定义优化函数optimizer_func
def optimizer_func(lr):
    return fluid.optimizer.Adagrad(learning_rate=lr)


# 定义数据提供器
def reader_creator(data, word_dict, tag_dict):
    UNK = word_dict['<unk>']
    INS = []

    with codecs.open(data, "r", encoding="utf-8") as f:
        for line in f:
            l = line.strip().split("\t")
            INS.append(([word_dict.get(w, UNK) for w in l[2]], tag_dict[l[1]]))

    def reader():
        for doc, label in INS:
            yield doc, label
    return reader

# 该函数用来计算训练中模型在test数据集上的结果
def train_test(program, reader):
    count = 0
    feed_var_list = [program.global_block().var(var_name) for var_name in feed_order]
    feeder_test = fluid.DataFeeder(feed_list=feed_var_list, place=place)
    test_exe = fluid.Executor(place)
    accumulated = len([avg_cost, accuracy]) * [0]
    for test_data in reader():
        avg_cost_np = test_exe.run(
            program=program,
            feed=feeder_test.feed(test_data),
            fetch_list=[avg_cost, accuracy])
        accumulated = [x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)]
        count += 1
    return [x / count for x in accumulated]



if __name__ == "__main__":

    CLASS_DIM = 44    # 情感分类的类别数
    EMB_DIM = 128     # 词向量的维度
    HID_DIM = 512     # 隐藏层的维度
    BATCH_SIZE = 256  # batch的大小
    STACKED_NUM = 3   #LSTM双向栈的层数
    lr = 0.002        # 学习率

    train_data = "./data/train_data.txt"                         # 训练数据
    test_data = "./data/test_data.txt"                           # 测试数据
    test_data_predict = "./data/test_data_predict.txt"           # 测试数据的预测结果，用于模型最终评估
    dict_data = './data/dict.txt'                                # 字典数据
    tag_data = './data/tags.txt'                                 # 类别
    params_dirname = "eligibilitycriteria_lstm.inference.model"  # 保存模型。参数

    train = False   # 控制是否进行训练
    predict = False # 进行预测
    evaluate = True # 在测试集上进行评估

    # 载入字典
    print("Loading Eligibility Criteria word dict......")
    with open(dict_data, 'r', encoding='utf-8') as f_dict:
        word_dict = eval(f_dict.readlines()[0])
    print("word dict length：{}".format(len(word_dict)))

    # 载入类别
    print("Loading Eligibility Criteria category tags......")
    with open(tag_data, 'r', encoding='utf-8') as f_tag:
        tag_dict = eval(f_tag.readlines()[0])
    print("category number：{}".format(len(tag_dict)))

    # 载入训练数据
    print ("Reading training data......")
    train_reader = fluid.io.batch(fluid.io.shuffle(reader_creator(train_data, word_dict, tag_dict), buf_size=25000), batch_size=BATCH_SIZE)

    # 载入测试数据
    print ("Reading testing  data......")
    test_reader = fluid.io.batch(reader_creator(test_data, word_dict, tag_dict), batch_size=BATCH_SIZE)

    # 进行训练
    if train == True:
        # 选择CPU进行训练
        use_cuda = True  #在cpu上进行训练
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

        # 构造训练器, 训练器需要一个训练程序和一个训练优化函数。
        exe = fluid.Executor(place)
        prediction = inference_program(word_dict)
        [avg_cost, accuracy] = train_program(prediction) #训练程序
        sgd_optimizer = optimizer_func(lr) #训练优化函数
        sgd_optimizer.minimize(avg_cost)

        # 提供数据并构建主训练循环
        # feed_order用来定义每条产生的数据和fluid.layers.data之间的映射关系。比如，imdb.train产生的第一列的数据对应的是words这个特征。
        feed_order = ['words', 'label']
        pass_num = 30  #训练循环的轮数

        #程序主循环部分
        def train_loop():
            #启动上文构建的训练器
            exe.run(fluid.default_startup_program()) # 进行参数初始化

            feed_var_list_loop = [fluid.default_main_program().global_block().var(var_name) for var_name in feed_order]
            feeder = fluid.DataFeeder(feed_list=feed_var_list_loop, place=place)
            
            # 获取预测程序
            test_program = fluid.default_main_program().clone(for_test=True)

            #训练循环
            for epoch_id in range(pass_num):
                for step_id, data in enumerate(train_reader()):
                    #运行训练器  
                    metrics = exe.run(fluid.default_main_program(), feed=feeder.feed(data), fetch_list=[avg_cost, accuracy])

                    #在测试集上测试结果
                    avg_cost_test, acc_test = train_test(test_program, test_reader)
                    print('Step {0}, Test Loss {1:0.2}, Acc {2:0.2}'.format(step_id, avg_cost_test, acc_test)) # 测试集上的评估评估
                    print("Step {0}, Epoch {1} Metrics {2}".format(step_id, epoch_id, list(map(np.array,metrics)))) # 训练集上的评估评估

            if params_dirname is not None:
                fluid.io.save_inference_model(params_dirname, ["words"], prediction, exe)#保存模型
            return
        # 开始训练
        train_loop()
        

    # 进行预测
    if predict == True:
        # 选择CPU进行预测
        use_cuda = False  #在cpu上进行预测
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        
        # 单条测试用输入数据
        criteria_str = [u'年龄大于18岁', u'性别不限', u'过去经常酗酒']
        criterias = [[w for w in c] for c in criteria_str]

        # 构建预测器
        exe = fluid.Executor(place)
        inference_scope = fluid.core.Scope()

        UNK = word_dict['<unk>']
        lod = []
        for c in criterias:
            lod.append([word_dict.get(words, UNK) for words in c])

        base_shape = [[len(c) for c in lod]]
        lod = np.array(sum(lod, []), dtype=np.int64)

        tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
        
        # 应用模型并进行预测
        with fluid.scope_guard(inference_scope):
            [inferencer, feed_target_names, fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

            assert feed_target_names[0] == "words"
            results = exe.run(inferencer,
                              feed={feed_target_names[0]: tensor_words},
                              fetch_list=fetch_targets,
                              return_numpy=False)
            np_data = np.array(results[0])
            for i, probability in enumerate(np_data):
                prob = probability.tolist()
                category = list(tag_dict.keys())[prob.index(max(prob))]
                print("the maximum predict probability of category for eligibility criteria sentence: [{}] is [{}]".format(criteria_str[i], category))


    # 在测试集上进行评估
    if evaluate == True:
        # 读取测试数据
        # 选择CPU进行预测
        use_cuda = False  #在cpu上进行预测
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

        # 读取测试数据，生成预测结果，用于评估
        with codecs.open(test_data, "r", encoding="utf-8") as f:
            criteria_data = [line.strip().split("\t") for line in f]
        criteria_str = [d[2] for d in criteria_data]
        criterias = [[w for w in c] for c in criteria_str]

        # 构建预测器
        exe = fluid.Executor(place)
        inference_scope = fluid.core.Scope()

        UNK = word_dict['<unk>']
        lod = []
        for c in criterias:
            lod.append([word_dict.get(words, UNK) for words in c])

        base_shape = [[len(c) for c in lod]]
        lod = np.array(sum(lod, []), dtype=np.int64)

        tensor_words = fluid.create_lod_tensor(lod, base_shape, place)
        
        # 应用模型并进行预测
        predict_category = []
        with fluid.scope_guard(inference_scope):
            [inferencer, feed_target_names, fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

            assert feed_target_names[0] == "words"
            results = exe.run(inferencer,
                              feed={feed_target_names[0]: tensor_words},
                              fetch_list=fetch_targets,
                              return_numpy=False)
            np_data = np.array(results[0])
            for i, probability in enumerate(np_data):
                prob = probability.tolist()
                category = list(tag_dict.keys())[prob.index(max(prob))]
                predict_category.append(category)
        with codecs.open(test_data_predict, "w", encoding="utf-8") as outf:
            for i in range(len(predict_category)):
                outf.write("{}\t{}\t{}\r\n".format(criteria_data[i][0], predict_category[i], criteria_data[i][2]))
        
        # 终端输入命令进行评估：
        # python evaluation.py ./data/test_data.txt ./data/test_data_predict.txt > ./data/test_data_evaluation.txt
