import numpy as np
#import random
from MVMT import Reg_MVMT

def main():
    view_0 = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    view_1 = np.matrix([[11, 12], [13, 14], [15, 16]])
    view_2 = np.matrix([[21, 22, 23], [24, 25, 26], [27, 28, 29]])

    tasks = {0 : [0, 1], 1 : [1, 2]}
    views = {0 : view_0, 1 : view_1, 2 : view_2}
    task_lables = {0 : [-1.0, 1.0, 0.0], 1 : [1.0, -1.0, 0.0]}

    mvmt = Reg_MVMT(tasks, task_lables, views)
    print mvmt.run_mvmt()
    
def read_data(files):
    views = []
    for file_name in files:
        data = np.loadtxt(open(file_name, 'r'), delimiter=',', usecols=range(1, 65)).tolist()
        labels = np.loadtxt(open(file_name, 'r'), delimiter=',', usecols=range(0, 1), dtype=str).tolist()

        views.append((labels, data))
    return views

def mapper_init(files):
    labels = []
    views = read_data(files)
    for view in views:
        for i in range(len(view[0])):
            labels.append(view[0][i], (view[1][0], view[1][1][i]))
    return tuple(labels)

def reducer(labels):
    samples = {}
    for label in labels:
        samples[label] = [[], []]
        
    for label in labels:
        view = label[1][0]
        features = label[1][1]
        label_id = label[0]
        
        samples[label_id][0].append(view)
        samples[label_id][1].append(features)
        
    return samples
    
def test():
    task_views = {}
    for i in range(100):
        task_views[i] = [0, 1, 2]
    views = {0 : '100_leaves_plant_species/data_Mar_64.txt',
             1 : '100_leaves_plant_species/data_Sha_64.txt',
             2 : '100_leaves_plant_species/data_Tex_64.txt'}
    
    samples = reducer(mapper_init())

    for v in views.keys():
    	views[v] = []
    	for i in samples.keys():
    		for j in samples[i][0]
    			if v == j:


    task_lables = {}
    for i in range(100):
    	task_lables[i] = samples[i]
    
    mvmt = Reg_MVMT(task_views, task_lables, views)
    W_t, Omega_v = mvmt.train()
    mvmv.predict(W_t, Omega_v, test_data)
    

if __name__ == '__main__':
    # main()
    test()