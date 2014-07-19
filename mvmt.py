'''
regMVMT

Define:
T -> # of Tasks
V -> # of Views
v -> specific view in [1 : V]
t -> specific task in [1 : T]
N - set of labeled data
M - set of unlabeled data
D - set of examples in V views

Input:
{y_t}_t=1_T -> labeled example column vector -> [0:n_t] -> {1, -1}
{X_t}_t=1_T -> labeled feature matrix -> [0:n_t, D_v]
{U_t}_v=1_V -> weights -> [u_1_v, ..., u_T_v] -> [0:D_v, 0:T]
lambda - regularization parameters
mu - coupling parameter
gamma - regularization parameter to penalize difference in view mapping function

Algorithm:
X_t_v - feature matrix N_t x D_v for labeled examples
U_t_v - feature matrix M_t x D_v for unlabeled examples
y_t -  example labels vector N_t x 1 {1, -1}
I_d - indicator matrix T x V,
    I_d(t, v) = 0 if v is missing from t
    otherwise I_d(t, v) = 1
f_t_v - learning function maps D_v -> {1, -1}
w_t_v - weight
A_tv -
B_vv'_t -
C_t'v - 
Omega - similarity matrix T x T between tasks
L - matrix T*D x T*D
R - column vector of E_1,1 to E_T,V

Output:
{W_t}_t=1_T - matrix where column t_th = w_t
{Omega_v}_v=1_V - task similarity matrix
'''

import numpy as np

class Reg_MVMT(object):
    def __init__(self, tasks, views):
        self.tasks = tasks # {task_key : example_values}
        self.views = views # {view_key : feature_matrix}

    def run_mvmt(self, iterations, lambda_parameter, mu, gamma):
        task_count = len(self.tasks.keys()) # V
        view_count = len(self.views.keys()) # V
        Omega = {}
        W = {}
        labeled_feature_matrices = {} # X {view_key : labeled_feature_matrix}
        unlabeled_feature_matrices = {} # U {view_key : unlabeled_feature_matrix}
        weights = {} # w_t_v {(task_key, view_key) : weight}
        examples = {} # {example_key : labeled or unlabeled}
        for example_index in len(self.tasks[1]):
            if self.tasks[1][example_index] != 0.0:
                examples[example_index] = 'labeled'
            else:
                examples[example_index] = 'unlabeled'
        for view_key in self.views.keys():
            labeled_feature_matrix = []
            unlabeled_feature_matrix = []
            for example_index in len(self.views[view_key].shape[0]):
                if examples[example_index] == 'labeled':
                    labeled_feature_matrix.append(self.views[view_key][example_index].tolist()[0])
                if examples[example_index] == 'unlabeled':
                    unlabeled_feature_matrix.append(self.views[view_key][example_index].tolist()[0])
            labeled_feature_matrices[view_key] = np.matrix(labeled_feature_matrix)
            unlabeled_feature_matrices[view_key] = np.matrix(unlabeled_feature_matrix)
        
        for iteration in range(iterations):
            A_tv = {}
            E_tv = {} # E_tv {(task_key, view_key)}
            B_VV_T = {}
            
            for (task_key, view_key) in [(t, self.views.keys()[v - 1]) for v in [t for t in self.tasks.keys()]]:
                # construct A_tv
                labeled_feature_matrix = labeled_feature_matrices[view_key]
                unlabeled_feature_matrix = unlabeled_feature_matrices[view_key]
                A_tv = lambda_parameter + (mu * (view_count - 1) * unlabeled_feature_matrix.T * unlabeled_feature_matrix) + \
                        ((labeled_feature_matrix.T * labeled_feature_matrix) / (view_count ** 2))
                
                # construct E_tv
                labels = np.matrix([x for x in self.tasks[task_key] if x != 0.0]).T
                E_tv[(task_key, view_key)] = (labeled_feature_matrix.T * labels) / view_count
                
                # construct B_vv'_t
                B_vv_t = {} # B_vv'_t {(view_key, view_key_other) : feature matrix}
                for view_key_other in [x for x in self.views.keys() if x != view_key]:
                    labeled_feature_matrix_other = labeled_feature_matrices[view_key_other]
                    unlabeled_feature_matrix_other = unlabeled_feature_matrices[view_key_other]
                    B_vv_t[(view_key, view_key_other)] = ((labeled_feature_matrix.T * labeled_feature_matrix_other) / (view_count ** 2)) - \
                                                            (mu * unlabeled_feature_matrix.T * unlabeled_feature_matrix_other)
                
                # construct C_t'v
                C_tv = gamma * c * I_Dv
        
        # construct L
        L = np.zeros((task_count * view_count, task_count * view_count))
        for i in range(task_count * view_count):
            for j in ((task_count * view_count) / 2):
                if i == j:
                    L[i, j] = 
        
        # construct R
        R = []
        for task_key in self.tasks.keys():
            for view_key in self.views.keys():
                R.extend(E_tv[(task_key, view_key)].T.tolist()[0])
        R = np.matrix(R).T
        
        # compute W
        W = L.I * R
        
        # update Omega_v