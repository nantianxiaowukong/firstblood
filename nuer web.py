# -*- coding: UTF-8 -*-
#Author : Alian
#
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

#neural network class definiton
class neuralnetwork:

    # initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #初始化几点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #学习率
        self.lr = learningrate
        #输入与隐含层之间的权重
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        #隐含层与输出层之间的权重
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        #激活函数
        self.active_function = lambda x: scipy.special.expit(x)
        pass

    #train the neural network
    def train(self,inputs_list,targets_list):
        # 把输入列表和目标值转为列矩阵
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        # 计算隐含层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐含层输出
        hidden_outputs = self.active_function(hidden_inputs)
        # 计算输出层输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出
        final_outputs = self.active_function(final_inputs)
        #计算输出误差
        outputs_errors = targets - final_outputs
        #计算隐含层误差
        hidden_errors = np.dot(self.who.T,outputs_errors)
        #更新隐含层与输出层之间的权重
        self.who += self.lr*np.dot(outputs_errors*final_outputs*(1.0-final_outputs),np.transpose(hidden_outputs))
        # 更新隐含层与输出层之间的权重
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs),np.transpose(inputs))
        pass

    #query the neural network
    def query(self,inputs_list):
        #把输入列表转为列矩阵
        inputs = np.array(inputs_list,ndmin=2).T
        #计算隐含层的输入
        hidden_inputs = np.dot(self.wih,inputs)
        #计算隐含层输出
        hidden_outputs = self.active_function(hidden_inputs)
        #计算输出层输入
        final_inputs = np.dot(self.who,hidden_outputs)
        #计算输出层的输出
        final_outputs = self.active_function(final_inputs)
        return final_outputs
#定义输入数据特征化函数
def factorizeinput(list,outputnodes):
    #把数据读入，并分隔开
    all_values = list.split(',')
    #把训练数据缩小到0.01-1范围内
    inputs = (np.asfarray(all_values[1:])/255.0*0.99) + 0.01
    #结果特征化,零矩阵加0.01
    targets = np.zeros(outputnodes) + 0.01
    #对应的数字设为0.99
    targets[int(all_values[0])] = 0.99
    return inputs,targets
#设置节点和学习率
inputnodes = 784
outputnodes = 10
hiddennodes = 100
learningrate = 0.3

#神经网络实例化
n = neuralnetwork(inputnodes,hiddennodes,outputnodes,learningrate)
#导入训练数据(小样本，所以一次性导入，若是大样本，则要循环读入，采用readline）
train_data_file = open(r'E:\develop project\project\learn\ml\mnist_train.csv','r')
train_data_list = train_data_file.readline()

#导入测试数据
test_data_file = open(r'E:\develop project\project\learn\ml\mnist_test.csv','r')
test_data_list = test_data_file.readline()

#训练
count = 0
scorecard = []
while list(train_data_list):
    shuru,jieguo = factorizeinput(train_data_list,outputnodes)
    n.train(shuru,jieguo)
    count += 1
    print("第 %d 次训练" % count)
    train_data_list = train_data_file.readline()
    pass
train_data_file.close()
while list(test_data_list):
    zhenshishuzi = test_data_list.split(',')[0]
    shuru, jieguo = factorizeinput(test_data_list, outputnodes)
    yuceshuzi = n.query(shuru)
    yuce = list(yuceshuzi).index(max(list(yuceshuzi)))
    print("真实值为：",zhenshishuzi)
    print("预测值为：",yuce)
    if int(yuce) == int(zhenshishuzi):
        scorecard.append(1)
    else:
        scorecard.append(0)
    test_data_list = test_data_file.readline()
    pass
test_data_file.close()
print("正确率为：",sum(scorecard)/len(scorecard))


