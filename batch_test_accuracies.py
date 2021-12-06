import pickle
import json
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import argparse

plt.rcParams.update({'figure.figsize':(16, 9), 'figure.dpi':100})

acc_file_1 = open('Batch_1000/test_eval_metrics/lr_3.txt', "r")
acc_list_1 = acc_file_1.readlines()
acc_list_1 = [float(i) for i in acc_list_1]

acc_file_2 = open('Batch_2000/test_eval_metrics/lr_3.txt', "r")
acc_list_2 = acc_file_2.readlines()
acc_list_2 = [float(i) for i in acc_list_2]

acc_file_3 = open('Batch_2500/test_eval_metrics/lr_3.txt', "r")
acc_list_3 = acc_file_3.readlines()
acc_list_3 = [float(i) for i in acc_list_3]

acc_file_4 = open('Batch_3000/test_eval_metrics/lr_3.txt', "r")
acc_list_4 = acc_file_4.readlines()
acc_list_4 = [float(i) for i in acc_list_4]

acc_file_5 = open('Batch_4000/test_eval_metrics/lr_3.txt', "r")
acc_list_5 = acc_file_5.readlines()
acc_list_5 = [float(i) for i in acc_list_5]

acc_file_6 = open('Batch_5000/test_eval_metrics/lr_3.txt', "r")
acc_list_6 = acc_file_6.readlines()
acc_list_6 = [float(i) for i in acc_list_6]

iters=[i for i in range (1,len(acc_list_1)+1)]

plt.plot(iters, acc_list_1, label='Batch size - 1000')
plt.plot(iters, acc_list_2, label='Batch size - 2000')
plt.plot(iters, acc_list_3, label='Batch size - 2500')
plt.plot(iters, acc_list_4, label='Batch size - 3000')
plt.plot(iters, acc_list_5, label='Batch size - 4000')
plt.plot(iters, acc_list_6, label='Batch size - 5000')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title("SGD")
plt.legend()
	
img_file = open('./batch_accuracy_SGD', "wb+")
plt.savefig(img_file)
plt.clf()

acc_file_1 = open('Batch_1000/test_eval_metrics/mnb_3.txt', "r")
acc_list_1 = acc_file_1.readlines()
acc_list_1 = [float(i) for i in acc_list_1]

acc_file_2 = open('Batch_2000/test_eval_metrics/mnb_3.txt', "r")
acc_list_2 = acc_file_2.readlines()
acc_list_2 = [float(i) for i in acc_list_2]

acc_file_3 = open('Batch_2500/test_eval_metrics/mnb_3.txt', "r")
acc_list_3 = acc_file_3.readlines()
acc_list_3 = [float(i) for i in acc_list_3]

acc_file_4 = open('Batch_3000/test_eval_metrics/mnb_3.txt', "r")
acc_list_4 = acc_file_4.readlines()
acc_list_4 = [float(i) for i in acc_list_4]

acc_file_5 = open('Batch_4000/test_eval_metrics/mnb_3.txt', "r")
acc_list_5 = acc_file_5.readlines()
acc_list_5 = [float(i) for i in acc_list_5]

acc_file_6 = open('Batch_5000/test_eval_metrics/mnb_3.txt', "r")
acc_list_6 = acc_file_6.readlines()
acc_list_6 = [float(i) for i in acc_list_6]

iters=[i for i in range (1,len(acc_list_1)+1)]

plt.plot(iters, acc_list_1, label='Batch size - 1000')
plt.plot(iters, acc_list_2, label='Batch size - 2000')
plt.plot(iters, acc_list_3, label='Batch size - 2500')
plt.plot(iters, acc_list_4, label='Batch size - 3000')
plt.plot(iters, acc_list_5, label='Batch size - 4000')
plt.plot(iters, acc_list_6, label='Batch size - 5000')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title("MNB")
plt.legend()
	
img_file = open('./batch_accuracy_MNB', "wb+")
plt.savefig(img_file)
plt.clf()


acc_file_1 = open('Batch_1000/test_eval_metrics/pac_3.txt', "r")
acc_list_1 = acc_file_1.readlines()
acc_list_1 = [float(i) for i in acc_list_1]

acc_file_2 = open('Batch_2000/test_eval_metrics/pac_3.txt', "r")
acc_list_2 = acc_file_2.readlines()
acc_list_2 = [float(i) for i in acc_list_2]

acc_file_3 = open('Batch_2500/test_eval_metrics/pac_3.txt', "r")
acc_list_3 = acc_file_3.readlines()
acc_list_3 = [float(i) for i in acc_list_3]

acc_file_4 = open('Batch_3000/test_eval_metrics/pac_3.txt', "r")
acc_list_4 = acc_file_4.readlines()
acc_list_4 = [float(i) for i in acc_list_4]

acc_file_5 = open('Batch_4000/test_eval_metrics/pac_3.txt', "r")
acc_list_5 = acc_file_5.readlines()
acc_list_5 = [float(i) for i in acc_list_5]

acc_file_6 = open('Batch_5000/test_eval_metrics/pac_3.txt', "r")
acc_list_6 = acc_file_6.readlines()
acc_list_6 = [float(i) for i in acc_list_6]

iters=[i for i in range (1,len(acc_list_1)+1)]

plt.plot(iters, acc_list_1, label='Batch size - 1000')
plt.plot(iters, acc_list_2, label='Batch size - 2000')
plt.plot(iters, acc_list_3, label='Batch size - 2500')
plt.plot(iters, acc_list_4, label='Batch size - 3000')
plt.plot(iters, acc_list_5, label='Batch size - 4000')
plt.plot(iters, acc_list_6, label='Batch size - 5000')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title("PAC")
plt.legend()
	
img_file = open('./batch_accuracy_PAC', "wb+")
plt.savefig(img_file)
plt.clf()



