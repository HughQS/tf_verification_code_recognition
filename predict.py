import numpy as np  
from PIL import Image  
import os  
import tensorflow as tf
import csv

MODEL_SAVE_PATH = './model/'


def get_meta_file(path=MODEL_SAVE_PATH):
    meta_file = None
    for file_name in os.listdir(path):
        if file_name.endswith(".meta"):
            meta_file = os.path.join(MODEL_SAVE_PATH, file_name)
    return meta_file


def model_predict(testData):  
    labelList = []
    total = len(testData)
    batch_size = 100
    #加载graph
    meta_file = get_meta_file(MODEL_SAVE_PATH)
    saver = tf.train.import_meta_graph(meta_file)
    graph = tf.get_default_graph()  
    #从graph取得 tensor，他们的name是在构建graph时定义的(查看上面第2步里的代码)  
    input_holder = graph.get_tensor_by_name("data-input:0")  
    keep_prob_holder = graph.get_tensor_by_name("keep-prob:0")  
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")  
    with tf.Session() as sess:  
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))  
        end_index = 0
        for i in range(total // batch_size):
            start_index = i * batch_size
            end_index = (i+1) * batch_size
            predict = sess.run(predict_max_idx, feed_dict={input_holder: testData[start_index:end_index, :, :, :], keep_prob_holder: 1.0})
            
            predictValue = np.squeeze(predict)
            predictValue = predictValue.tolist()
            for row in predictValue:
                code = "".join([str(digit) for digit in row])
                labelList.append(code)
    return labelList


def save_as_csv(resultlist, filePath="./submission.csv"):
    """保存submission.csv文件"""
    submission = []
    size = len(resultlist)
    print(size)
    for index in range(size):
        submission.append([str(index+1),resultlist[index]])
    print("Saving csv...")
    #headers_train = ["id","label"]
    with open(filePath, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f,delimiter=',')
        #writer.writerow(headers_train)
        writer.writerows(submission)
    print("All done")


if __name__ == '__main__': 
    print("Reading image")
    test_data = np.stack([(np.array(Image.open("./test/" + str(index) +".jpg")))/255.0 for index in range(1, 10001, 1)])
    print("Loaded model,predicting")
    test_label = model_predict(test_data)
    save_as_csv(test_label)

