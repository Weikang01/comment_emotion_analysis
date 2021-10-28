"""
project description:
input->vocab embedding->fc->lstm->sequence_pooling->fc->output
                          ->---sequence_pooling---->
"""
import os
import pandas as pd
from time import time

# region path_definition
st_time = time()
project_root_dir = './emotion_analysis_for_hotel_comment/'
if not os.path.exists(project_root_dir):
    os.makedirs(project_root_dir)
raw_data_path = project_root_dir + 'hotel_discuss2.csv'
dict_save_path = project_root_dir + 'comment_dict.txt'
test_file_path = project_root_dir + 'test_file.txt'
train_file_path = project_root_dir + 'train_file.txt'
model_save_dir = project_root_dir + 'model/'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('path definition complete. time:', time() - st_time)
# endregion
#####################################################################

# region data_import_and_dict_construction
st_time = time()
raw_data = pd.read_csv(raw_data_path, delimiter=',', header=None, names=['opinion', 'comment'], encoding='utf-8')
raw_data['comment'].map(lambda x: x.replace('\n', ''))

print(raw_data.head(2))

char_set = set()
for comment in raw_data['comment']:
    for char in comment:
        char_set.add(char)

char_dict = {}
i = 0
for char in char_set:
    char_dict[char] = i
    i += 1
char_dict['<unk>'] = i

with open(dict_save_path, 'w', encoding='utf-8'):
    pass

with open(dict_save_path, 'a', encoding='utf-8') as f:
    f.write(str(char_dict))

print('dict construction complete. time:', time() - st_time)
# endregion
#####################################################################

# region text_encoding_and_test/train_file_creation
st_time = time()


def comment_encoding(comment_seq):
    return_list = [0] * len(comment_seq)
    i = 0
    for title in comment_seq:
        char_list = []
        for char in title:
            if char in char_dict:
                char_list.append(str(char_dict[char]))
            else:
                char_list.append(str(char_dict['<unk>']))
        return_list[i] = ','.join(char_list)
        i += 1
    return return_list


raw_data['encoded_comment'] = comment_encoding(raw_data['comment'])
raw_data.drop('comment', axis=1)

with open(test_file_path, 'w'):
    pass
with open(train_file_path, 'w'):
    pass

test_file = open(test_file_path, 'a', encoding='utf-8')
train_file = open(train_file_path, 'a', encoding='utf-8')

for i in range(len(raw_data)):
    line = f'{raw_data.loc[i, "encoded_comment"]}\t{raw_data.loc[i, "opinion"]}\n'
    if not i % 10:
        test_file.write(line)
    else:
        train_file.write(line)

test_file.close()
train_file.close()

print('data pre-processing completed. time:', time() - st_time)
# endregion
#####################################################################

# region nn_related_models_import
st_time = time()
import numpy as np
import paddle
import paddle.fluid as fluid
from multiprocessing import cpu_count

print('nn related models import completed. time:', time() - st_time)
# endregion
#####################################################################

# region reader_creation
st_time = time()


def data_mapper(sample):
    data, lab = sample
    val = [int(code) for code in data.split(',') if code.is_digit()]
    return val, int(lab)


def train_reader(train_file_path):
    def reader():
        with open(train_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            np.random.shuffle(lines)
            for line in lines:
                data, lab = line.split('\t')
            yield data, lab

    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)


print('reader construction complete. time:', time() - st_time)
# endregion
#####################################################################

# region lstm_network_definition
st_time = time()


def lstm_net(input_data, input_dimension):
    fluid.layers.reshape(input_data, [-1, 1], inplace=True)
    emb = fluid.layers.embedding(input_data, size=[input_dimension, 128], is_sparse=True)

    fc1 = fluid.layers.fc(emb, size=128)

    lstm, _ = fluid.layers.dynamic_lstm(fc1, size=128)

    conv1 = fluid.layers.sequence_pool(lstm, pool_type='max')
    conv2 = fluid.layers.sequence_pool(fc1, pool_type='max')

    out = fluid.layers.fc([conv1, conv2], size=2, act='softmax')
    return out


print('lstm function definition complete. time:', time() - st_time)
# endregion
#####################################################################

# region variables_and_optimizer_definition
st_time = time()

d_decoded = fluid.layers.data('data', [1], dtype='int64', lod_level=1)
label = fluid.layers.data('label', [1], dtype='int64')

predict = lstm_net(d_decoded, len(char_dict))

cost = fluid.layers.cross_entropy(label=label, input=predict)
avg_cost = fluid.layers.mean(cost)

acc = fluid.layers.accuracy(predict, label, 1)

optimizer = fluid.optimizer.AdamOptimizer(0.0001)
optimizer.minimize(avg_cost)
print('region variables/optimizer creation complete. time:', time() - st_time)
# endregion
#####################################################################

# region training
st_time = time()
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(program=fluid.default_startup_program())

BATCH_SIZE = 128  # do not set this value too high if you are not very confident to your computer capacity

t_reader = train_reader(train_file_path)
batch_reader = paddle.batch(t_reader, BATCH_SIZE)

feeder = fluid.DataFeeder([d_decoded, label], place)

epochs = 3

for epoch in range(epochs):
    for batch_id, data in enumerate(batch_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(), feed=feeder.feed(data),
                                        fetch_list=[avg_cost, acc])
        if not 20 % batch_id:
            print(f'train epoch: {epoch}--train batch:{batch_id}--train cost:{train_cost[0]}--accuracy:{train_acc[0]}')
print('training complete. time:', time() - st_time)
# endregion
#####################################################################

# region model saving
for f in os.listdir(model_save_dir):
    os.remove(model_save_dir + f)
st_time = time()
fluid.io.save_inference_model(model_save_dir, feeded_var_names=[d_decoded.name], target_vars=[predict], executor=exe)
print('training model saved. time:', time() - st_time)
# endregion
#####################################################################

# region test prediction
test_text_list = [
    "总体来说房间非常干净,卫浴设施也相当不错,交通也比较便利",
    "酒店交通方便，环境也不错，正好是我们办事地点的旁边，感觉性价比还可以",
    "设施还可以，服务人员态度也好，交通还算便利",
    "酒店服务态度极差，设施很差",
    "我住过的最不好的酒店,以后决不住了",
    "说实在的我很失望，我想这家酒店以后无论如何我都不会再去了"
]

lods = []
for comment in test_text_list:
    ret = []
    for char in comment:
        if char not in char_dict:
            char = '<unk>'
        ret.append(char_dict[char])
    lods.append(ret)

place = fluid.CPUPlace()
infer_exe = fluid.Executor(place)
infer_exe.run(fluid.default_startup_program())

base_shape = [[len(c) for c in lods]]

tensor_words = fluid.create_lod_tensor(lods, base_shape, place)

infer_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(dirname=model_save_dir,
                                                                                executor=infer_exe)
results = infer_exe.run(program=infer_program, feed={feed_target_names[0]:tensor_words}, fetch_list=fetch_targets)

for i, r in enumerate(results[0]):
    print(f'{test_text_list[i]} -- neg:{r[0]} // pos:{r[1]}')

# endregion
