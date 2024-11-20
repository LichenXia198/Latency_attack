import xlrd
import numpy as np
from datetime import date,datetime
import pylab as plt
from scipy.stats import norm

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


file = 'faster-rcnn.xlsx'

def norm_distrbution(x):

	mu =np.mean(x) #计算均值
	print(mu)
	sigma =np.std(x) 
	num_bins = 100 #直方图柱子的数量 
	n, bins, patches = plt.hist(x, num_bins,density=1, alpha=0.75) 
	#直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象 
	y = norm.pdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y

	plt.grid(True)
	plt.plot(bins, y, 'r--') #绘制y的曲线 
	plt.xlabel('values') #绘制x轴 
	plt.ylabel('Probability') #绘制y轴 
	plt.title('Histogram : $\mu$=' + str(round(mu,2)) + ' $\sigma=$'+str(round(sigma,2)))  #中文标题 u'xxx' 
	#plt.subplots_adjust(left=0.15)#左边距 
	plt.show()

	return y 

def read_excel():

	wb = xlrd.open_workbook(filename=file)#打开文件

	# print(wb.sheet_names())#获取所有表格名字
	# sheet1 = wb.sheet_by_index(0)#通过索引获取表格

	sheet2 = wb.sheet_by_name('city10-wp-threshold')#通过名字获取表格

	resize = sheet2.col_values(1)[1:]
	preprocess = sheet2.col_values(2)[1:]
	backbone = sheet2.col_values(3)[1:]
	rpn = sheet2.col_values(4)[1:]
	roi_pooling =  sheet2.col_values(5)[1:]
	postprocess = sheet2.col_values(6)[1:]
	read_img = sheet2.col_values(7)[1:]
	inference_nms = sheet2.col_values(8)[1:]
	drawbox = sheet2.col_values(9)[1:]
	e2e = sheet2.col_values(10)[1:]

	proposals = sheet2.col_values(11)[1:]
	box = sheet2.col_values(12)[1:]

	# print(roi_pooling)
	# norm_distrbution(resize)
	# norm_distrbution(preprocess)
	# norm_distrbution(backbone)
	# norm_distrbution(rpn)
	# norm_distrbution(roi_pooling)
	# norm_distrbution(postprocess)
	t0 = np.mean(resize) + np.mean(preprocess) + np.mean(backbone) + np.mean(rpn) + np.mean(roi_pooling) + np.mean(postprocess) + np.mean(read_img)
	print(t0)
	# print(np.mean(inference_nms))
	a = [resize, preprocess, backbone, rpn, roi_pooling, postprocess, read_img, inference_nms, drawbox, e2e, proposals, box]
	# print(e2e)

	# print(sheet1,sheet2)
	# print(sheet1.name,sheet1.nrows,sheet1.ncols)
	# rows = sheet1.row_values(2)#获取行内容
	# cols = sheet1.col_values(3)#获取列内容
	# print(rows)
	# print(cols)
	# print(sheet1.cell(1,0).value)#获取表格里的内容，三种方式
	# print(sheet1.cell_value(1,0))
	# print(sheet1.row(1)[0].value)
	return a

t = read_excel()
