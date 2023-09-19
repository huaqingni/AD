from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import matlab.engine
from sklearn import metrics
#from sklearn.metrics import roc_auc_score
#from playsound import playsound
import tensorflow_addons as tfa

# standard code
# triple losses
# no attention
# 2021-09-14
# 2022-06-10

tf.keras.backend.set_floatx('float32')
eng = matlab.engine.start_matlab()
loss_object = tf.keras.losses.MeanSquaredError()
loss_object2 = tf.keras.losses.SparseCategoricalCrossentropy()
loss_object3 = tfa.losses.TripletSemiHardLoss()

optimizer = tf.keras.optimizers.Adam()

'''
# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)
'''

# gpu setting
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# encode network
def encode():
    inputs = tf.keras.layers.Input(shape=[input_shape, ])  # shape=(110,)??    输入向量
    layer0 = tf.keras.layers.Flatten(dtype='float32')  # 数据偏平化
    layer3 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')  # 全连接层，he_normal：初始核的数值设置，He分布
    layer4 = tf.keras.layers.ReLU()



    x = layer0(inputs)
    y = layer3(x)
    y = layer4(y)


    return tf.keras.Model(inputs=inputs, outputs=y)


# decode network
def decode():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])
    layer0 = tf.keras.layers.Flatten(dtype='float32')


    layer1 = tf.keras.layers.Dense(input_shape, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.Reshape(target_shape=(input_shape, 1))

    x = layer0(inputs)

    y = layer1(x)
    y = layer2(y)
    return tf.keras.Model(inputs=inputs, outputs=y)


# residual_block for classification of hidden feature
def residual_block(filters, apply_dropout=True):
    result = tf.keras.Sequential()  # 采用sequential构造法
    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # l2正则化权重0.1
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))

    result.add(tf.keras.layers.ReLU())

    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))  # l2正则化权重0.1
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))

    result.add(tf.keras.layers.ReLU())
    return result


# classification network
def classify():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])

    block_stack_1 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]  # residual_block为block_stack的元素！
    block_stack_2 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_3 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]

    layer0 = tf.keras.layers.Flatten(dtype='float32')
    layer_in = tf.keras.layers.Dense(num_of_hidden_classify, kernel_initializer='he_normal', activation='relu')
    layer_out = tf.keras.layers.Dense(4, kernel_initializer='he_normal', activation='softmax')

    res_x_0 = 0
    res_x_1 = 0
    res_x_2 = 0

    x = inputs
    x = layer0(x)  # output dimension: num_of_hidden
    x = layer_in(x)  # output dimension: num_of_hidden_classify

    x_0 = x
    for block in block_stack_1:
        res_x_0 = block(x)  # 实现一次residual_block调用
    x = res_x_0 + x  # 体现残差网络

    for block in block_stack_2:
        res_x_1 = block(x)
    x = res_x_1 + x  # 在过一次residual_block

    for block in block_stack_3:
        res_x_2 = block(x)
    x = res_x_2 + x  # 在过一次residual_block，相当于深度特征

    x = x_0 + x  # 将当于特征增强
    x = layer_out(x)  # 分类输出, output dimension: 2
    return tf.keras.Model(inputs=inputs, outputs=x)


'''
# 分类时的聚类性能代价函数（暂时未用）
def loss_fn(data, labels):  # 评价函数，非代价函数
    # yh0_np = np.array(data)
    yh0_np = data
    if labels:
        pass
    # h0_label = tf.reshape(labels, (len(labels),))

    yh0_np_true_avg = tf.reduce_mean(yh0_np, axis=(0,))
    yh0_np_false_avg = tf.reduce_mean(yh0_np, axis=(0,))
    y0_h0_h1_distance = tf.norm(yh0_np_true_avg - yh0_np_false_avg)

    yh0_distance_true = tf.reduce_mean(tf.norm((yh0_np - yh0_np_true_avg), axis=1, keepdims=True))
    yh0_distance_false = tf.reduce_mean(tf.norm((yh0_np - yh0_np_false_avg), axis=1, keepdims=True))

    return -y0_h0_h1_distance, yh0_distance_true, yh0_distance_false'''

@tf.function
# 代价函数定义和梯度求导
def train_step(images, labels, encoder, decoder, classifier):
    with tf.GradientTape() as encode_tape, tf.GradientTape() as decode_tape, tf.GradientTape() as classify_tape, tf.GradientTape() as funsion_tape:
        y = encoder(images)

        z = decoder(y)
        predicted_label = classifier(y)

        loss1 = loss_object(images, z)
        loss2 = loss_object2(labels, predicted_label)   #多分类loss
        loss3 = 0.1 * loss_object3(labels, y)
        loss3 = 0.0* tfa.losses.TripletSemiHardLoss(margin=2.0)(labels, y)   # 看看能不能用


        loss_sum = loss1 + loss2 + loss3

    gradient_e = encode_tape.gradient(loss_sum, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_e, encoder.trainable_variables))

    gradient_d = decode_tape.gradient(loss1, decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_d, decoder.trainable_variables))

    gradient_c = classify_tape.gradient(loss2, classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradient_c, classifier.trainable_variables))

    return loss1, loss2,  loss3,predicted_label


# 数据准备,训练数据已排序，ADHD在前，HC在后
def prepare_data(index, tag):
    # 处理 Alff数据

    train_h_data, train_h_label ,tag_label,testlabel = eng.svm_four_class(
        index, tag, nargout=4)  # 调用matlab时，一般默认返回参数只有一个，这里说明返回参数有6个

    train_h_label = np.array(train_h_label)
    train_h_label = train_h_label.reshape((len(train_h_label),))

    train_h_data = np.array(train_h_data)
    tag_label = np.array(tag_label)
    tag_label = np.int32(tag_label.reshape((len(tag_label),)))


    return train_h_data,  train_h_label, tag_label, testlabel
# H1下训练模型
def train_h1(train_data, train_label, print_information=True):

    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, input_shape))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1, loss2, loss3, predicted_label = train_step(images=train_x, labels=label_x, encoder=encoder, decoder=decoder,
                                                   classifier=classifier)

        train_accuracy(train_label, predicted_label)
        if print_information and epoch_count==EPOCH-1:
            template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, Loss3 triple loss: {}, Accuracy : {}%, {}%, {}% {}%'
            loss1_account_for = 100 * loss1 / (loss1 + loss2 + loss3)
            loss2_account_for = 100 * loss2 / (loss1 + loss2 + loss3)
            loss3_account_for = 100 * loss3 / (loss1 + loss2 + loss3)
            print(template2.format(epoch_count, loss1, loss2, loss3,
                                   train_accuracy.result() * 100,
                                   loss1_account_for,
                                   loss2_account_for,
                                   loss3_account_for
                                   )
                  )

    y_h1_x = encoder(train_data)

    # tf.keras.backend.clear_session()
    # del encoder, decoder, classifier
    return y_h1_x, train_label


def train_h2(train_data, train_label, print_information=True):

    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, input_shape))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1, loss2, loss3, predicted_label = train_step(images=train_x, labels=label_x, encoder=encoder, decoder=decoder,
                                                   classifier=classifier)

        train_accuracy(train_label, predicted_label)
        if print_information and epoch_count==EPOCH-1:
            template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, Loss3 triple loss: {}, Accuracy : {}%, {}%, {}% {}%'
            loss1_account_for = 100 * loss1 / (loss1 + loss2 + loss3)
            loss2_account_for = 100 * loss2 / (loss1 + loss2 + loss3)
            loss3_account_for = 100 * loss3 / (loss1 + loss2 + loss3)
            print(template2.format(epoch_count, loss1, loss2, loss3,
                                   train_accuracy.result() * 100,
                                   loss1_account_for,
                                   loss2_account_for,
                                   loss3_account_for
                                   )
                  )

    y_h2_x = encoder(train_data)

    # tf.keras.backend.clear_session()
    # del encoder, decoder, classifier
    return y_h2_x, train_label

def train_h3(train_data, train_label, print_information=True):

    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, input_shape))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1, loss2, loss3, predicted_label = train_step(images=train_x, labels=label_x, encoder=encoder, decoder=decoder,
                                                   classifier=classifier)

        train_accuracy(train_label, predicted_label)
        if print_information and epoch_count==EPOCH-1:
            template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, Loss3 triple loss: {}, Accuracy : {}%, {}%, {}% {}%'
            loss1_account_for = 100 * loss1 / (loss1 + loss2 + loss3)
            loss2_account_for = 100 * loss2 / (loss1 + loss2 + loss3)
            loss3_account_for = 100 * loss3 / (loss1 + loss2 + loss3)
            print(template2.format(epoch_count, loss1, loss2, loss3,
                                   train_accuracy.result() * 100,
                                   loss1_account_for,
                                   loss2_account_for,
                                   loss3_account_for
                                   )
                  )

    y_h3_x = encoder(train_data)

    # tf.keras.backend.clear_session()
    # del encoder, decoder, classifier
    return y_h3_x, train_label

def train_h4(train_data, train_label, print_information=True):

    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, input_shape))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1, loss2, loss3, predicted_label = train_step(images=train_x, labels=label_x, encoder=encoder, decoder=decoder,
                                                   classifier=classifier)

        train_accuracy(train_label, predicted_label)
        if print_information and epoch_count==EPOCH-1:
            template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, Loss3 triple loss: {}, Accuracy : {}%, {}%, {}% {}%'
            loss1_account_for = 100 * loss1 / (loss1 + loss2 + loss3)
            loss2_account_for = 100 * loss2 / (loss1 + loss2 + loss3)
            loss3_account_for = 100 * loss3 / (loss1 + loss2 + loss3)
            print(template2.format(epoch_count, loss1, loss2, loss3,
                                   train_accuracy.result() * 100,
                                   loss1_account_for,
                                   loss2_account_for,
                                   loss3_account_for
                                   )
                  )

    y_h3_x = encoder(train_data)

    # tf.keras.backend.clear_session()
    # del encoder, decoder, classifier
    return y_h3_x, train_label




def Mymeasure(y_h0_x, tag_label):
    yh0_np = np.array(y_h0_x)  # h0时的特征数据
    tag_ = tag_label.cumsum()
    #yh0_HC, yh0_PD, yh0_AC, yh0_GD = np.split(yh0_np, tag_label)
    yh0_PD = yh0_np[0:tag_[0]]
    yh0_AC = yh0_np[tag_[0]:tag_[1]]
    yh0_GD = yh0_np[tag_[1]:tag_[2]]
    yh0_4 = yh0_np[tag_[2]:tag_[3]]


    # 类内/类间距离

    yh0_PD_avg = np.mean(yh0_PD, axis=(0,))  # h0 HC均值
    yh0_AC_avg = np.mean(yh0_AC, axis=(0,))  # h0 ADHD均值
    yh0_GD_avg = np.mean(yh0_GD, axis=(0,))  # h0 HC均值
    yh0_4_avg= np.mean(yh0_4, axis=(0,))
    yh0_all_avg = np.mean(yh0_np, axis=(0,))  # 总均值


    yh0_intra_PD = np.sum(np.power(np.linalg.norm((yh0_PD - yh0_PD_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_AC = np.sum(np.power(np.linalg.norm((yh0_AC - yh0_AC_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_GD = np.sum(np.power(np.linalg.norm((yh0_GD - yh0_GD_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_4 = np.sum(np.power(np.linalg.norm((yh0_4 - yh0_4_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_all = yh0_intra_PD +  yh0_intra_AC + yh0_intra_GD+yh0_intra_4


    yh0_inter_PD = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_PD_avg), axis=0, keepdims=True), 2))
    yh0_inter_AC = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_AC_avg), axis=0, keepdims=True), 2))
    yh0_inter_GD = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_GD_avg), axis=0, keepdims=True), 2))
    yh0_inter_4 = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_4_avg), axis=0, keepdims=True), 2))
    yh0_inter_all = tag_label[0] * yh0_inter_PD +  tag_label[
        1] * yh0_inter_AC + tag_label[2] * yh0_inter_GD+tag_label[3] * yh0_inter_4

    yh0_out_class = yh0_intra_all / yh0_inter_all

    return yh0_out_class



# for PD, konggao, jiaolv classification
def judge3(y_h1_x, y_h2_x, y_h3_x, y_h4_x,tag_label):


    yh1_out_class = Mymeasure(y_h1_x, tag_label)
    yh2_out_class = Mymeasure(y_h2_x, tag_label)
    yh3_out_class = Mymeasure(y_h3_x, tag_label)
    yh4_out_class = Mymeasure(y_h4_x, tag_label)


    tmp_ = [ yh1_out_class, yh2_out_class, yh3_out_class,yh4_out_class]
    min_v = min(tmp_)
    min_index = tmp_.index(min_v)+1

    return min_index




def train():


    predict = []
    true= []
    m=0
    for i in range(dict_data[name_of_data]):
        train_h1_data, train_label ,tag_label,testlabel = prepare_data(
            index=i + 109 , tag=1)
        train_h2_data, train_label, tag_label, testlabel = prepare_data(
            index=i + 109,tag=2)
        train_h3_data, train_label, tag_label, testlabel = prepare_data(
            index=i + 109,tag=3)
        train_h4_data, train_label, tag_label, testlabel = prepare_data(
            index=i + 109,tag=4)


        train_label=train_label-1
        y_h1_x, _ = train_h1(train_h1_data, train_label, print_information=False)
        tf.keras.backend.clear_session()
        y_h2_x, _ = train_h1(train_h2_data, train_label, print_information=False)
        tf.keras.backend.clear_session()
        y_h3_x, _ = train_h1(train_h3_data, train_label, print_information=False)
        tf.keras.backend.clear_session()
        y_h4_x, _ = train_h1(train_h4_data, train_label, print_information=False)
        tf.keras.backend.clear_session()


        judge_result2 = judge3( y_h1_x, y_h2_x, y_h3_x,y_h4_x,tag_label)    # for other disease classification

        if  judge_result2  == testlabel:
            m += 1
        print('\n current loop:' + str(i + 1) + ' / ' + str(dict_data[name_of_data]) + '-------------')
        print('-------------' + str(j_out + 1) + ' / ' + '50' + '-------------\n')
        print('current accuracy: ' + str(m) + '/' + str(i + 1))
        predict.append(judge_result2)

    return predict

if __name__ == '__main__':
    name_list = ['ASD', 'Peking_data', 'KKI_data', 'NI_data', 'Peking_1_data']
    dict_data = {'ASD': 287, 'Peking_data': 194, 'KKI_data': 83, 'NI_data': 48, 'Peking_1_data': 86}
    EPOCH_list = {'ASD': 100, 'Peking_data': 100, 'KKI_data': 50, 'NI_data': 35, 'Peking_1_data': 50}



    for i_out in range(0,1):

        name_of_data = name_list[i_out]
        input_shape = 60 #可调参数
        num_of_hidden = 30  # 自编码网络隐层特征数，可调参数
        num_of_hidden_classify = 20  # 分类器隐层特征数，可调参数
        EPOCH = EPOCH_list[name_of_data]  # Epoch，可调参数

        Batch_size = 178

        '''logs = '# input shape:' + str(input_shape) + ',hidden: ' + str(num_of_hidden) + ',' + \
               str(num_of_hidden_classify) + ',epoch: ' + str(EPOCH) + ' \n'
        with open('./newresults/' + name_of_data + '.txt', "a+") as f:
            f.write(logs)'''

        for j_out in range(3):

            encoder = encode()
            decoder = decode()
            classifier = classify()

            out_1 = train()

            del encoder, decoder, classifier



