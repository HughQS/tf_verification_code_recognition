# -*- coding: utf-8 -*-
import numpy as np
import os
import time
import random
from PIL import Image
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CODE_LEN = 5
CHAR_SET_LEN = 10
IMAGE_WIDHT = 200
IMAGE_HEIGHT = 60
MODEL_SAVE_PATH = './model/'
TRAIN_IMAGE_PATH = './train'

TRAIN_IMAGE_PERCENT = 1

def get_image_file_name(imgPath=TRAIN_IMAGE_PATH):
    txt_file = os.path.join(TRAIN_IMAGE_PATH, "train.csv")
    images_path = []
    images_labels = []
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split(',')
            images_path.append(item[0] + ".jpg")
            images_labels.append(item[1])
    return images_path, images_labels

def name2label(name):
    label = np.zeros(CODE_LEN * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i*CHAR_SET_LEN + ord(c) - ord('0')
        label[idx] = 1
    return label


# 取得验证码图片数据和它里面的数字
def get_data_and_label(fileName, label, filePath=TRAIN_IMAGE_PATH):
    pathName = os.path.join(filePath, fileName)
    img = Image.open(pathName)
    image_array = np.array(img)
    image_data = image_array/255.0
    image_label = name2label(label)
    return image_data, image_label


#生成一个训练batch
def get_next_batch(image_name_list, image_label_list, train_batch_size, step):
    batch_data = np.zeros([train_batch_size, IMAGE_HEIGHT, IMAGE_WIDHT, 3])
    batch_label = np.zeros([train_batch_size, CODE_LEN * CHAR_SET_LEN])   

    totalNumber = len(image_name_list)
    indexStart = step*train_batch_size
    for i in range(train_batch_size):
        index = (i + indexStart) % totalNumber
        name = image_name_list[index]
        label = image_label_list[index]
        img_data, img_label = get_data_and_label(name, label)
        batch_data[i, :, :, :] = img_data
        batch_label[i, :] = img_label
    return batch_data, batch_label


#构建卷积神经网络并训练
def train_data_with_CNN(train_image_name, train_image_label, train_batch_size=64,
                         valid_image_name=None, valid_image_label=None, valid_batch_size=64, total_step=20000):
    #初始化权值
    def weight_variable(shape, name='weight'):
        init = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var
    #初始化偏置
    def bias_variable(shape, name='bias'):
        init = tf.constant(0.1, shape=shape)
        var = tf.Variable(init, name=name)
        return var
    #卷积
    def conv2d(x, W, name='conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
    #池化
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    #输入层
    #请注意 X 的 name，在测试model时会用到它
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDHT, 3], name='data-input')
    Y = tf.placeholder(tf.float32, [None, CODE_LEN * CHAR_SET_LEN], name='label-input')
    x_input = tf.reshape(X, [-1, IMAGE_HEIGHT, IMAGE_WIDHT, 3], name='x-input')
    #dropout,防止过拟合
    #请注意 keep_prob 的 name，在测试model时会用到它
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    #第一层卷积
    W_conv1 = weight_variable([3, 3, 3, 16], 'W_conv1')
    B_conv1 = bias_variable([16], 'B_conv1')

    conv1 = conv2d(x_input, W_conv1, 'conv1')
    _mean, _var = tf.nn.moments(conv1, [0, 1, 2])
    conv1 = tf.nn.batch_normalization(conv1, _mean, _var, 0, 1, 0.0001)
    conv1 = tf.nn.relu(conv1 + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    #第二层卷积
    W_conv2 = weight_variable([3, 3, 16, 32], 'W_conv2')
    B_conv2 = bias_variable([32], 'B_conv2')

    conv2 = conv2d(conv1, W_conv2,'conv2')
    _mean, _var = tf.nn.moments(conv2, [0, 1, 2])
    conv2 = tf.nn.batch_normalization(conv2, _mean, _var, 0, 1, 0.0001)
    conv2 = tf.nn.relu(conv2 + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    #第三层卷积
    W_conv3 = weight_variable([3, 3, 32, 32], 'W_conv3')
    B_conv3 = bias_variable([32], 'B_conv3')

    conv3 = conv2d(conv2, W_conv3, 'conv3')
    _mean, _var = tf.nn.moments(conv3, [0, 1, 2])
    conv3 = tf.nn.batch_normalization(conv3, _mean, _var, 0, 1, 0.0001)
    conv3 = tf.nn.relu(conv3 + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    #全链接层
    #每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    W_fc1 = weight_variable([25*8*32, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, 25*8*32])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    #输出层
    W_fc2 = weight_variable([1024, CODE_LEN * CHAR_SET_LEN], 'W_fc2')
    B_fc2 = bias_variable([CODE_LEN * CHAR_SET_LEN], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    predict = tf.reshape(output, [-1, CODE_LEN, CHAR_SET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, CODE_LEN, CHAR_SET_LEN], name='labels')
    #预测结果
    #请注意 predict_max_idx 的 name，在测试model时会用到它
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))

    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for steps in range(total_step):
            train_data, train_label = get_next_batch(train_image_name, train_image_label, train_batch_size, steps)
            sess.run(optimizer, feed_dict={X : train_data, Y : train_label, keep_prob:0.75})
            if steps % 100 == 0:
                test_data, test_label = get_next_batch(valid_image_name, valid_image_label, valid_batch_size, steps)
                acc, my_loss, my_predict_max_idx = sess.run([accuracy, loss, predict_max_idx], feed_dict={X : test_data, Y : test_label, keep_prob:1.0})
                print("steps=%d, loss=%f, accuracy=%f" % (steps, my_loss, acc))
                #print(my_predict_max_idx)
                if acc > 0.99:
                    saver.save(sess, MODEL_SAVE_PATH+"tf_verification_code.model", global_step=steps)
            


if __name__ == '__main__':
    train_batch_size = 64
    valid_batch_size = 512
    total_step = 60000
    image_filename_list, image_label_list = get_image_file_name(TRAIN_IMAGE_PATH)
    total = len(image_filename_list)
    random.seed(time.time())
    #打乱顺序
    data_class_list = list(zip(image_filename_list, image_label_list))  # zip压缩合并，将数据与标签对应压缩
    random.shuffle(data_class_list)
    trainImageNumber = int(total * TRAIN_IMAGE_PERCENT)
    train_list = data_class_list[:trainImageNumber]  # 训练集
    #test_list = data_class_list[trainImageNumber:]  # 测试集
    train_image_name, train_image_label = zip(*train_list)  # 训练集解压缩
    #valid_image_name, valid_image_label = zip(*test_list)  # 测试集解压缩
    train_data_with_CNN(train_image_name, train_image_label, train_batch_size,
                         train_image_name, train_image_label, valid_batch_size, total_step)
    print('Training finished')
