from mrjob.job import MRJob
from mrjob.step import MRStep

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
        (((task_key, view_key)), ((sample_values, feature_matrix), (W, Omega, W0, Omega0)))

# return W and Omega
map:
    (((task_key, view_key)), ((sample_values, feature_matrix), (W, Omega, W0, Omega0)))
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

    '''
    def configure_options(self):
        super(MVMT_Job, self).configure_options()

        self.add_passthrough_option(
            '-e', '--views', type='str', default=None,
            help=('Files for each view. Required.'))

        self.add_passthrough_option(
            '-e', '--tasks', type='int', default=1,
            help=('The number of tasks. Required.'))

    '''

    '''
    def read_data(self, files):
        views = []
        for file_name in files:
            data = np.loadtxt(open(file_name, 'r'), delimiter=',', usecols=range(1, 65)).tolist()
            labels = np.loadtxt(open(file_name, 'r'), delimiter=',', usecols=range(0, 1), dtype=str).tolist()

            views.append((labels, data))
        return views

    '''

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

    '''
    def reduce_views(self, label_key, view_features):
        for view, features in view_features:
            yield(view, features)

    '''

    '''
    def construct_A_E_B_C(self, (task, view), ()):


    def steps(self):
        return [MRStep(mapper_init=self.map_views, reducer=self.reduce_views),
                MRStep(reducer=self.construct_A_E_B_C)]
        
if __name__ == '__main__':
    task_views = {}
    for i in range(100):
        tasks[i] = [0, 1, 2]
    views = {0 : '100_leaves_plant_species\\data_Mar_64.txt',
             1 : '100_leaves_plant_species\\data_Sha_64.txt',
             2 : '100_leaves_plant_species\\data_Tex_64.txt'}

    mvmt = MVMT_Job(task_views, views)

    mvmt.run()