import pickle
import json
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import argparse


parser = argparse.ArgumentParser(description='Plots graphs for given batch size folder')
parser.add_argument('--batch_num', '-bn', help='Batch Size Folder to Use', required=True, type=str)

args = parser.parse_args()
folder=args.batch_num

folder_name='./'+folder+'/'

plt.rcParams.update({'figure.figsize':(16, 9), 'figure.dpi':100})

i='lr'
file_name_1 = i+'_'+'1'+'.txt'
file_name_2 = i+'_'+'2'+'.txt'
file_name_3 = i+'_'+'3'+'.txt'
acc_file_1 = open(folder_name+'test_eval_metrics/'+file_name_1, "r")
acc_list_1 = acc_file_1.readlines()
acc_list_1 = ['{:.2f}'.format(float(i)) for i in acc_list_1]
acc_file_2 = open(folder_name+'test_eval_metrics/'+file_name_2, "r")
acc_list_2 = acc_file_2.readlines()
acc_list_2 = ['{:.2f}'.format(float(i)) for i in acc_list_2]
acc_file_3 = open(folder_name+'test_eval_metrics/'+file_name_3, "r")
acc_list_3 = acc_file_3.readlines()
acc_list_3 = ['{:.2f}'.format(float(i)) for i in acc_list_3]
iters=[i for i in range (1,len(acc_list_1)+1)]

plt.plot(iters,acc_list_1,label='loss function -log');
plt.plot(iters,acc_list_2,label='loss function -hinge');
plt.plot(iters,acc_list_3,label='loss function -perceptron');
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title("SGD Classifier")
plt.legend()
	
img_file = open(folder_name + i, "wb+")
plt.savefig(img_file)
plt.clf()


i='mnb'
file_name_1 = i+'_'+'1'+'.txt'
file_name_2 = i+'_'+'2'+'.txt'
file_name_3 = i+'_'+'3'+'.txt'
acc_file_1 = open(folder_name+'test_eval_metrics/'+file_name_1, "r")
acc_list_1 = acc_file_1.readlines()
acc_list_1 = ['{:.2f}'.format(float(i)) for i in acc_list_1]
acc_file_2 = open(folder_name+'test_eval_metrics/'+file_name_2, "r")
acc_list_2 = acc_file_2.readlines()
acc_list_2 = ['{:.2f}'.format(float(i)) for i in acc_list_2]
acc_file_3 = open(folder_name+'test_eval_metrics/'+file_name_3, "r")
acc_list_3 = acc_file_3.readlines()
acc_list_3 = ['{:.2f}'.format(float(i)) for i in acc_list_3]
iters=[i for i in range (1,len(acc_list_1)+1)]

plt.plot(iters,acc_list_1,label='parameter alpha = 1.0');
plt.plot(iters,acc_list_2,label='parameter alpha = 0.5');
plt.plot(iters,acc_list_3,label='parameter alpha = 0.7');
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title("Multinomial Naive Bayes Classifier")
plt.legend()
	
img_file = open(folder_name + i, "wb+")
plt.savefig(img_file)
plt.clf()



i='pac'
file_name_1 = i+'_'+'1'+'.txt'
file_name_2 = i+'_'+'2'+'.txt'
file_name_3 = i+'_'+'3'+'.txt'
acc_file_1 = open(folder_name+'test_eval_metrics/'+file_name_1, "r")
acc_list_1 = acc_file_1.readlines()
acc_list_1 = ['{:.2f}'.format(float(i)) for i in acc_list_1]
acc_file_2 = open(folder_name+'test_eval_metrics/'+file_name_2, "r")
acc_list_2 = acc_file_2.readlines()
acc_list_2 = ['{:.2f}'.format(float(i)) for i in acc_list_2]
acc_file_3 = open(folder_name+'test_eval_metrics/'+file_name_3, "r")
acc_list_3 = acc_file_3.readlines()
acc_list_3 = ['{:.2f}'.format(float(i)) for i in acc_list_3]
iters=[i for i in range (1,len(acc_list_1)+1)]

plt.plot(iters,acc_list_1,label='parameter C = 0.2');
plt.plot(iters,acc_list_2,label='parameter C = 0.5');
plt.plot(iters,acc_list_3,label='parameter C = 1.0');
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title("Passive Aggressive Classifier")
plt.legend()
	
img_file = open(folder_name + i, "wb+")
plt.savefig(img_file)
plt.clf()




