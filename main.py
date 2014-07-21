import numpy as np
from random import shuffle
from mvmt import Reg_MVMT

def test():
	view_0 = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	view_1 = np.matrix([[11, 12], [13, 14], [15, 16]])
	view_2 = np.matrix([[21, 22, 23], [24, 25, 26], [27, 28, 29]])

	tasks = {0 : [0, 1], 1 : [1, 2]}
	views = {0 : view_0, 1 : view_1, 2 : view_2}
	task_labels = {0 : [-1.0, 1.0, 0.0], 1 : [1.0, -1.0, 0.0]}

	mvmt = Reg_MVMT(tasks, task_labels, views)
	mvmt.run_mvmt()

def read_data(files):
	views = []
	for file_name in files:
		data = np.loadtxt(open(file_name, 'r'), delimiter=',', usecols=range(1, 65)).tolist()
		labels = np.loadtxt(open(file_name, 'r'), delimiter=',', usecols=range(0, 1), dtype=str).tolist()

		views.append((labels, data))
	return views

def learn(tasks, views):
	files = views.values()
	data = read_data(files)
	views_count = len(views.keys())
	views = {}
	task_lables = {}
	train_data = {}
	test_data = {}

	for view in data:
		data[v] = random.shuffle(data[v])
		for i in range(len(data[v]):
			if i < int(len(data[v]) * 0.3):
				test_data[data[v][i][0]] = data[v][i][1]
			else:
				train_data[data[v][i][0]] = data[v][i][1]
				task_lables = 

	for t in tasks.keys():
		task_lables = 

	for v in range(views_count):
		view[v] = 


	mvmt = Reg_MVMT(tasks, task_lables, views)
	mvmt.run_mvmt()

if __name__ == '__main__':
	tasks = {}
	for i in range(100):
		tasks[i] = [0, 1, 2]
	views = {0 : '100_leaves_plant_species\\data_Mar_64.txt',
			 1 : '100_leaves_plant_species\\data_Sha_64.txt',
			 2 : '100_leaves_plant_species\\data_Tex_64.txt'}

	learn(task, views)