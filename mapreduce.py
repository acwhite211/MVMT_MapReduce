from mrjob.job import MRJob
from mrjob.step import MRStep
from random import shuffle

'''
MVMT_Job Outline

# setup tasks and views, also initialize W, Omega, W0, and Omega0
map: 
    (task_views, files)
    ->*
    ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0)))

for i in range(iterations):

    # construct A, E, B, C
    map:
        ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0)))
        ->
        ('mvmt', ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0), (A, E, B, C)))

    # construct R and L, then compute W and Omega
    reduce:
        ('mvmt', ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0), (A, E, B, C)))
        ->*
        ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0)))

# return W and Omega
map:
    ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0)))
    ->
    ('mvmt', (W, Omega))

reduce:
    ('mvmt', (W, Omega))
    ->
    (W, Omega)
'''
class MVMT_Job(MRJob):

    def __init__(self, task_views,  views):
        self.task_views = task_views # {task_key : view_keys}
        self.tasks = {} # {task_key : sample_values}
        self.views = views # {view_key : feature_matrix}

    '''
    command line arguments
    '''
    def configure_options(self):
        super(MVMT_Job, self).configure_options()

        self.add_passthrough_option(
            '-e', '--views', type='str', default=None,
            help=('Files for each view. Required.'))

        self.add_passthrough_option(
            '-e', '--tasks', type='int', default=1,
            help=('The number of tasks. Required.'))

    def read_data(self, files):
        views = []
        for file_name in files:
            data = np.loadtxt(open(file_name, 'r'), delimiter=',', usecols=range(1, 65)).tolist()
            labels = np.loadtxt(open(file_name, 'r'), delimiter=',', usecols=range(0, 1), dtype=str).tolist()

            views.append((labels, data))
        return views

    
    '''
    map: 
    (task_views, files)
    ->*
    ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0)))
    '''
    def map_views(self):
        files = self.views.values()
        data = read_data(files)

        for view in data:
            for sample_index in range(len(view[0])):
                label = view[0][sample_index]
                features = view[1][sample_index]
                yield(label, (view, features))

    '''
    map:
    ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0)))
    ->
    ('mvmt', ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0), (A, E, B, C)))
    '''
    def map_construct(self, (task_key, view_key), ((sample_values, feature_matrix), (T, V, D, W, Omega, W0, Omega0, W_t))):
        t = task_key
        v = view_key

        A = {}
        B = {}
        C = {}
        E = {}

        # construct A[t, v]
        A[t, v] = lambda_var + (mu * (V - 1) * U[t, v].T * U[t, v]) + \
                  ((X[t, v].T * X[t, v]) / (V ** 2))

        # construct E[t, v]
        E[t, v] = (X[t, v].T * y[t]) / V

        # construct B[t, v, v']
        for v2 in range(V):
            if v != v2:
                B[t, v, v2] = ((X[t, v].T * X[t, v2]) / (V ** 2)) - \
                              (mu * U[t, v].T * U[t, v2])

        # construct C[t', v]
        for t2 in range(T):
            if t != t2:
                I_Dv = np.matrix(np.identity(self.views[v].shape[1]))
                C[t2, v] = gamma * Omega[v][t, t2] * I_Dv

        yield('mvmt', ((sample_values, feature_matrix), (T, V, D, W, Omega, W0, Omega0, W_t), (A, E, B, C)))

    '''
    reduce:
    ('mvmt', ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0), (A, E, B, C)))
    ->*
    ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0)))
    '''
    def reduce_construct(self, name_key, ((sample_values, feature_matrix), (T, V, D, W, Omega, W0, Omega0, W_t), (A, E, B, C))):
        # construct L
        L = np.zeros((T * D, T * D))
        i_offset = 0
        j_offset = 0
        for t in range(T):
            row_index = 0
            for v in range(V):
                for t2 in range(T):
                    if t == t2:
                        for v2 in range(V):
                            if v == v2:
                                for i in range(A[t, v].shape[0]):
                                    for j in range(A[t, v].shape[1]):
                                        L[i + i_offset, j + j_offset] = A[t, v][i, j]
                                j_offset += A[t, v].shape[1]
                            else:
                                for i in range(B[t, v, v2].shape[0]):
                                    for j in range(B[t, v, v2].shape[1]):
                                        L[i + i_offset, j + j_offset] = B[t, v, v2][i, j]
                                j_offset += B[t, v, v2].shape[1]
                    else:
                        for v2 in range(V):
                            if v == v2:
                                for i in range(C[t, v].shape[0]):
                                    for j in range(C[t, v].shape[1]):
                                        L[i + i_offset, j + j_offset] = C[t, v][i, j]
                                j_offset += C[t, v].shape[1]
                            else:
                                j_offset += C[t, v2].shape[1]
                i_offset += A[t, row_index].shape[0]
                j_offset = 0
                row_index += 1
        L = np.matrix(L)

        # construct R -> column vector
        R = []
        for t in range(T):
            for v in range(V):
                R.extend(E[(t, v)].T.tolist()[0])
        R = np.matrix(R).T

        # compute W
        W = L.I * R

        # construct W_v [V x T]
        W_v = []
        for v in range(V):
            for t in range(T):
                W_v.extend(W_t[t, v].T.tolist()[0])
        W_v = np.matrix(W_v).T

        # update Omega[v]
        for v in range(V):
            Omega[v] = ((W_v.T * W_v) ** (1 / 2)) / (sum(np.diag((W_v.T * W_v) ** (1 / 2)).tolist()))

        for t in range(T):
            for v in range(V):
                yield((t, v), ((sample_values, feature_matrix), (T, V, D, W, Omega, W0, Omega0, W_t)))

    '''
    map:
    ((task_key, view_key), ((sample_values, feature_matrix), (W, Omega, W0, Omega0)))
    ->
    ('mvmt', (W, Omega))
    '''
    def map_return(self, (task_key, view_key), ((sample_values, feature_matrix), (T, V, D, W, Omega, W0, Omega0, W_t))):
        yield(T, V, D, W, Omega, W_t)
        
    '''
    reduce:
    ('mvmt', (W, Omega))
    ->
    (W, Omega)
    '''
    def reduce_return(self, name_key, (T, V, D, W, Omega, W_t)):
        W = W[0]
        Omega = Omega[0]
        W_t = W_t[0]

        # reconstruct W_t
        W_t = W_t.T.tolist()
        W_temp = W.T.tolist()[0]
        for t in range(T):
            W_t[t] = W_temp[t * D : (t * D) + D]
        W_t = np.matrix(W_t).T
    
    def steps(self):
        job_procedure = []
        job_procedure.append(MRStep(mapper_init=self.map_views))
        for i in range(100):
            job_procedure.append(MRStep(mapper=self.map_construct, reducer=self.reduce_construct))
        job_procedure.append(MRStep(mapper=self.map_return, reducer=self.reduce_return))
        
        return job_procedure
        
if __name__ == '__main__':
    task_views = {}
    for i in range(100):
        tasks[i] = [0, 1, 2]
    views = {0 : '100_leaves_plant_species\\data_Mar_64.txt',
             1 : '100_leaves_plant_species\\data_Sha_64.txt',
             2 : '100_leaves_plant_species\\data_Tex_64.txt'}

    mvmt = MVMT_Job(task_views, views)

    mvmt.run()