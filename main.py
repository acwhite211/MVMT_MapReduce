import numpy as np

def main():
	view_0 = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
	view_1 = np.matrix([[11, 12], [13, 14], [15, 16]])
	view_2 = np.matrix([[21, 22, 23], [24, 25, 26], [27, 28, 29]])

	tasks = {0 : [0, 1], 1 : [1, 2]}
	views = {0 : view_0, 1 : view_1, 2 : view_2}
	task_lables = {0 : [-1.0, 1.0, 0.0], 1 : [1.0, -1.0, 0.0]}

	mvmt = Reg_MVMT(tasks, task_lables, views)
	mvmt.run_mvmt()

def read_data(files):
	views = []
	for file_name in files:
		data = np.loadtxt(open(file_name, 'r'), delimiter=',', usecols=range(1, 65)).tolist()
		labels = np.loadtxt(open(file_name, 'r'), delimiter=',', usecols=range(0, 1), dtype=str).tolist()
		views.append((labels, data))
	return views

if __name__ == '__main__':
	files = ['100_leaves_plant_species\\data_Mar_64.txt',
			 '100_leaves_plant_species\\data_Sha_64.txt',
			 '100_leaves_plant_species\\data_Tex_64.txt']
	print read_data(files)