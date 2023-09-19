from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import matlab.engine


eng = matlab.engine.start_matlab()
loss_object = tf.keras.losses.MeanSquaredError()
loss_object2 = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

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
    inputs = tf.keras.layers.Input(shape=[numf, ])      # input feature number = 50
    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer1 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.ReLU()

    x = layer0(inputs)
    y = layer1(x)
    y = layer2(y)
    return tf.keras.Model(inputs=inputs, outputs=y)

# decode network
def decode():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])
    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer1 = tf.keras.layers.Dense(numf, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.Reshape(target_shape=(numf, 1))

    x = layer0(inputs)
    y = layer1(x)
    y = layer2(y)
    return tf.keras.Model(inputs=inputs, outputs=y)

def subencode():
    inputs = tf.keras.layers.Input(shape=[input_shape, ])  # shape=(110,)??    输入向量
    layer0 = tf.keras.layers.Flatten(dtype='float32')  # 数据偏平化
    layer3 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')  # 全连接层，he_normal：初始核的数值设置，He分布
    layer4 = tf.keras.layers.ReLU()
    layer5 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')  # 全连接层，he_normal：初始核的数值设置，He分布
    layer6 = tf.keras.layers.ReLU()
    layer1 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')  # 全连接层，he_normal：初始核的数值设置，He分布
    layer2 = tf.keras.layers.ReLU()

    x = layer0(inputs)
    y = layer3(x)
    y = layer4(y)
    y = layer5(y)
    y = layer6(y)
    y = layer1(y)
    y = layer2(y)

    return tf.keras.Model(inputs=inputs, outputs=y)

# decode network
def subdecode():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])
    layer0 = tf.keras.layers.Flatten(dtype='float32')
    layer3 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')
    layer4 = tf.keras.layers.ReLU()
    layer5 = tf.keras.layers.Dense(num_of_hidden, kernel_initializer='he_normal')
    layer6 = tf.keras.layers.ReLU()
    layer1 = tf.keras.layers.Dense(input_shape, kernel_initializer='he_normal')
    layer2 = tf.keras.layers.Reshape(target_shape=(input_shape, 1))

    x = layer0(inputs)
    y = layer3(x)
    y = layer4(y)
    y = layer5(y)
    y = layer6(y)
    y = layer1(y)
    y = layer2(y)
    return tf.keras.Model(inputs=inputs, outputs=y)


# residual_block for classification of hidden feature
def residual_block(filters, apply_dropout=True):
    result = tf.keras.Sequential()  # 采用sequential构造法
    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())

    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())
    return result
def subresidual_block(filters, apply_dropout=True):
    result = tf.keras.Sequential()  # 采用sequential构造法
    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())

    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())
    return result
# classification network
def classify():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])

    block_stack_1 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_2 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_3 = [residual_block(num_of_hidden_classify, apply_dropout=True), ]

    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer_in = tf.keras.layers.Dense(num_of_hidden_classify, kernel_initializer='he_normal', activation='relu')
    layer_out = tf.keras.layers.Dense(2, kernel_initializer='he_normal', activation='softmax')

    res_x_0 = 0
    res_x_1 = 0
    res_x_2 = 0

    x = inputs
    x = layer0(x)
    x = layer_in(x)

    x_0 = x
    for block in block_stack_1:
        res_x_0 = block(x)
    x = res_x_0 + x

    for block in block_stack_2:
        res_x_1 = block(x)
    x = res_x_1 + x

    for block in block_stack_3:
        res_x_2 = block(x)
    x = res_x_2 + x

    x = x_0 + x
    x = layer_out(x)   # output dimension: 2
    return tf.keras.Model(inputs=inputs, outputs=x)
def subclassify():
    inputs = tf.keras.layers.Input(shape=[num_of_hidden, ])

    block_stack_1 = [subresidual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_2 = [subresidual_block(num_of_hidden_classify, apply_dropout=True), ]
    block_stack_3 = [subresidual_block(num_of_hidden_classify, apply_dropout=True), ]

    layer0 = tf.keras.layers.Flatten(dtype='float64')
    layer_in = tf.keras.layers.Dense(num_of_hidden_classify, kernel_initializer='he_normal', activation='relu')
    layer_out = tf.keras.layers.Dense(4, kernel_initializer='he_normal', activation='softmax')

    res_x_0 = 0
    res_x_1 = 0
    res_x_2 = 0

    x = inputs
    x = layer0(x)
    x = layer_in(x)

    x_0 = x
    for block in block_stack_1:
        res_x_0 = block(x)
    x = res_x_0 + x

    for block in block_stack_2:
        res_x_1 = block(x)
    x = res_x_1 + x

    for block in block_stack_3:
        res_x_2 = block(x)
    x = res_x_2 + x

    x = x_0 + x
    x = layer_out(x)   # output dimension: 4
    return tf.keras.Model(inputs=inputs, outputs=x)
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as encode_tape, tf.GradientTape() as decode_tape, tf.GradientTape() as classify_tape:
        y = encoder(images)
        z = decoder(y)
        predicted_label = classifier(y)

        loss1 = loss_object(images, z)
        loss2 = loss_object2(labels, predicted_label)
        loss_sum = loss1 + loss2

    gradient_e = encode_tape.gradient(loss_sum, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_e, encoder.trainable_variables))

    gradient_d = decode_tape.gradient(loss1, decoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_d, decoder.trainable_variables))

    gradient_c = classify_tape.gradient(loss2, classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradient_c, classifier.trainable_variables))

    return loss1, loss2, predicted_label

def subtrain_step(images, labels, subencoder, subdecoder, subclassifier):
    with tf.GradientTape() as subencode_tape, tf.GradientTape() as subdecode_tape, tf.GradientTape() as subclassify_tape:
        y = subencoder(images)
        z = subdecoder(y)
        predicted_label = subclassifier(y)

        loss1 = loss_object(images, z)
        loss2 = loss_object2(labels, predicted_label)
        loss_sum = loss1 + loss2

    gradient_e = subencode_tape.gradient(loss_sum, subencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_e, subencoder.trainable_variables))

    gradient_d = subdecode_tape.gradient(loss1, subdecoder.trainable_variables)
    optimizer.apply_gradients(zip(gradient_d, subdecoder.trainable_variables))

    gradient_c = subclassify_tape.gradient(loss2, subclassifier.trainable_variables)
    optimizer.apply_gradients(zip(gradient_c, subclassifier.trainable_variables))

    return loss1, loss2, predicted_label

def prepare_data(index):

    # get functional connections and their labels by matlab code
    train_h0_data, train_h0_label, train_h1_data, train_h1_label, test_h0_label, test_h1_label  ,testlabel= eng.svm_two_class(index, numf,nargout=7)

    train_h0_data = np.array(train_h0_data)
    train_h0_label = np.array(train_h0_label)
    train_h1_data = np.array(train_h1_data)
    train_h1_label = np.array(train_h1_label)
    test_h0_label = np.array(test_h0_label)
    test_h1_label = np.array(test_h1_label)
    testlabel = np.array(testlabel)

    num_h0 = train_h0_label.sum()  # ADHD subjects in h0 hypothesis
    num_h1 = train_h1_label.sum()  # ADHD subjects in h1 hypothesis

    return train_h0_data, train_h0_label, train_h1_data, train_h1_label, num_h0, num_h1, test_h0_label, test_h1_label,testlabel


def subprepare_data(index,tag):
    train_h_data, train_h_label, tag_label, testlabel = eng.svm_four_class(
        index, tag,input_shape, nargout=4)  # 调用matlab时，一般默认返回参数只有一个，这里说明返回参数有6个

    train_h_label = np.array(train_h_label)
    train_h_label = train_h_label.reshape((len(train_h_label),))

    train_h_data = np.array(train_h_data)
    tag_label = np.array(tag_label)
    tag_label = np.int32(tag_label.reshape((len(tag_label),)))

    return train_h_data, train_h_label, tag_label, testlabel


# h0 training model
def train_h(train_data, train_label, print_information=False):
    for epoch_count in range(EPOCH):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (Batch_size, numf))
        label_x = np.reshape(train_label, (Batch_size,))
        loss1, loss2, predicted_label = train_step(images=train_x, labels=label_x)

        train_accuracy(train_label, predicted_label)
        if print_information:
            template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, Accuracy : {}%, {}%, {}%'
            loss1_account_for = 100 * loss1 / (loss1 + loss2)
            loss2_account_for = 100 * loss2 / (loss1 + loss2)
            print(template2.format(epoch_count, loss1, loss2, train_accuracy.result() * 100, loss1_account_for,
                                   loss2_account_for))

    y_h0_x = encoder(train_data)
    tf.keras.backend.clear_session()
    return y_h0_x, train_label

def subtrain_h(train_data, train_label, print_information=False):
    for epoch_count in range(epoch):
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        train_x = tf.reshape(train_data, (batchsize, input_shape))
        label_x = np.reshape(train_label, (batchsize,))
        loss1, loss2, predicted_label = subtrain_step(images=train_x, labels=label_x, subencoder=subencoder, subdecoder=subdecoder,
                                                   subclassifier=subclassifier)

        train_accuracy(train_label, predicted_label)
        if print_information:
            template2 = 'Epoch : {} Loss1 mse: {}, Loss2 cross entropy: {}, Accuracy : {}%, {}%, {}%'
            loss1_account_for = 100 * loss1 / (loss1 + loss2)
            loss2_account_for = 100 * loss2 / (loss1 + loss2)
            print(template2.format(epoch_count, loss1, loss2, train_accuracy.result() * 100, loss1_account_for,
                                   loss2_account_for))

    y_h0_x = subencoder(train_data)
    tf.keras.backend.clear_session()
    return y_h0_x, train_label


def judge2(y_h0_x, h0_label, y_h1_x, h1_label, num_h0, num_h1):
    if h0_label is None:
        if h1_label is None:
            pass

    # h0
    yh0_np = np.array(y_h0_x)  # deeper feature in h0
    yh0_AD = np.split(yh0_np, (num_h0,))
    yh0_AD = np.array(yh0_AD)
    yh0_HC = np.copy(yh0_AD)
    yh0_AD = np.delete(yh0_AD, 1, axis=0)[0]
    yh0_HC = np.delete(yh0_HC, 0, axis=0)[0]

    # inter- and intra-class distance
    yh0_AD_avg = np.mean(yh0_AD, axis=(0,))
    yh0_HC_avg = np.mean(yh0_HC, axis=(0,))
    yh0_all_avg = np.mean(yh0_np, axis=(0,))

    yh0_intra_AD = np.sum(np.power(np.linalg.norm((yh0_AD - yh0_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_HC = np.sum(np.power(np.linalg.norm((yh0_HC - yh0_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh0_intra_all = yh0_intra_AD + yh0_intra_HC

    yh0_inter_AD = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_AD_avg), axis=0, keepdims=True), 2))
    yh0_inter_HC = np.sum(np.power(np.linalg.norm((yh0_all_avg - yh0_HC_avg), axis=0, keepdims=True), 2))
    yh0_inter_all = num_h0 * yh0_inter_AD + (yh0_np.shape[0] - num_h0) * yh0_inter_HC

    yh0_out_class = yh0_intra_all / yh0_inter_all

    # h1
    yh1_np = np.array(y_h1_x)  # deeper feature in h1
    yh1_AD = np.split(yh1_np, (num_h1,))
    yh1_AD = np.array(yh1_AD)
    yh1_HC = np.copy(yh1_AD)
    yh1_AD = np.delete(yh1_AD, 1, axis=0)[0]
    yh1_HC = np.delete(yh1_HC, 0, axis=0)[0]

    # inter- and intra-class distance
    yh1_AD_avg = np.mean(yh1_AD, axis=(0,))  # h1 ADHD均值
    yh1_HC_avg = np.mean(yh1_HC, axis=(0,))  # h1 HC均值
    yh1_all_avg = np.mean(yh1_np, axis=(0,))  # 总均值

    yh1_intra_AD = np.sum(np.power(np.linalg.norm((yh1_AD - yh1_AD_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_HC = np.sum(np.power(np.linalg.norm((yh1_HC - yh1_HC_avg), axis=1, keepdims=True).flatten(), 2))
    yh1_intra_all = yh1_intra_AD + yh1_intra_HC

    yh1_inter_AD = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_AD_avg), axis=0, keepdims=True), 2))
    yh1_inter_HC = np.sum(np.power(np.linalg.norm((yh1_all_avg - yh1_HC_avg), axis=0, keepdims=True), 2))
    yh1_inter_all = num_h1 * yh1_inter_AD + (yh1_np.shape[0] - num_h1) * yh1_inter_HC

    yh1_out_class = yh1_intra_all / yh1_inter_all

    # ADHD decision function
    if yh1_out_class >= yh0_out_class:
        return True
    else:
        return False

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

def judge4(y_h1_x, y_h2_x, y_h3_x, y_h4_x,tag_label):


    yh1_out_class = Mymeasure(y_h1_x, tag_label)
    yh2_out_class = Mymeasure(y_h2_x, tag_label)
    yh3_out_class = Mymeasure(y_h3_x, tag_label)
    yh4_out_class = Mymeasure(y_h4_x, tag_label)


    tmp_ = [ yh1_out_class, yh2_out_class, yh3_out_class,yh4_out_class]
    min_v = min(tmp_)
    min_index = tmp_.index(min_v)+1

    return min_index



def train():
    j = 0
    k = 0
    hc=0
    pred_real = []   # ground truth label
    m=0
    for i in range(dict_data[name_of_data]):
        train_h0_data, train_h0_label, train_h1_data, train_h1_label, num_h0, num_h1, test_h0_label, test_h1_label ,testlabel= prepare_data(
            index=i + 1)


        y_h0, train_label_h0 = train_h(train_h0_data, train_h0_label, print_information=False)
        tf.keras.backend.clear_session()
        y_h1, train_label_h1 = train_h(train_h1_data, train_h1_label, print_information=False)
        tf.keras.backend.clear_session()

        judge_result2 = judge2(y_h0, train_h0_label, y_h1, train_h1_label, num_h0, num_h1)

        if (judge_result2 ==True and testlabel==0)or(judge_result2 ==False and testlabel !=0):
            k += 1
        if judge_result2 == True :
            if testlabel==0:
                hc=hc+1
            pred_real.append(0)
        if judge_result2 ==False:

            train_h1_data, train_label, tag_label, testlabel = subprepare_data(
                index=i + 1, tag=1)
            train_h2_data, train_label, tag_label, testlabel = subprepare_data(
                index=i +1, tag=2)
            train_h3_data, train_label, tag_label, testlabel = subprepare_data(
                index=i + 1, tag=3)
            train_h4_data, train_label, tag_label, testlabel = subprepare_data(
                index=i + 1, tag=4)

            train_label = train_label - 1
            y_h1_x, _ = subtrain_h(train_h1_data, train_label,print_information=False)
            tf.keras.backend.clear_session()
            y_h2_x, _ = subtrain_h(train_h2_data, train_label, print_information=False)
            tf.keras.backend.clear_session()
            y_h3_x, _ = subtrain_h(train_h3_data, train_label, print_information=False)
            tf.keras.backend.clear_session()
            y_h4_x, _ = subtrain_h(train_h4_data, train_label, print_information=False)
            tf.keras.backend.clear_session()

            judge_result4 = judge4(y_h1_x, y_h2_x, y_h3_x, y_h4_x, tag_label)
            pred_real.append(judge_result4)
            if judge_result4 == testlabel:
                m += 1

        print('\n current loop:' + str(i+1) + ' / ' + str(dict_data[name_of_data]) + '-------------')
        print('-------------' + str(j_out+1) + ' / ' + '50' + '-------------\n')
        print('current accuracy: ' + str(hc+m) + '/' + str(i + 1))



    results_txt = str(k/287) + '\t'+str((hc+m)/287)+'\n'

    with open('./results/' + 'all'+name_of_data +str(ii)+'.txt', "a+") as f:
        f.write(results_txt)
    with open('./results/' + "all_pre_label" +str(ii)+ '.txt', "a+") as f:
        f.write(str(pred_real)+'\n')


if __name__ == '__main__':
    name_list = ['ASD', 'Peking_data', 'KKI_data', 'NI_data', 'Peking_1_data']
    dict_data = {'ASD': 287, 'Peking_data': 194, 'KKI_data': 83, 'NI_data': 48, 'Peking_1_data': 86}
    EPOCH_list = {'ASD': 100, 'Peking_data': 100, 'KKI_data': 50, 'NI_data': 35, 'Peking_1_data': 50}

    for i_out in range(0, 1):   # select ADHD-200 datasets
        for ii in [50]:
            numf=ii
            input_shape=ii
            name_of_data = name_list[i_out]
            num_of_hidden = 30              # neural unit in auto-coding network
            num_of_hidden_classify = 20     # neural unit in classification network
            Batch_size = dict_data[name_of_data]-1
            EPOCH = EPOCH_list[name_of_data]
            epoch=100
            batchsize = 178
            for j_out in range(10):
                encoder = encode()
                decoder = decode()
                classifier = classify()
                subencoder = subencode()
                subdecoder = subdecode()
                subclassifier = subclassify()

                train()

                del encoder
                del decoder
                del classifier
                del subencoder
                del subdecoder
                del subclassifier