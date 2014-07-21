from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol
import re
import numpy as np

WORD_RE = re.compile(r"[\w']+")


class Load_Files(MRJob):

    OUTPUT_PROTOCOL = mrjob.protocol.RawValueProtocol
    
    def __init__(self, files):
        self.views = files

    def read_data(self, files):
        views = []
        for file_name in files.keys():
            data = np.loadtxt(open(files[file_name], 'r'), delimiter=',', usecols=range(1, 65)).tolist()
            labels = np.loadtxt(open(files[file_name], 'r'), delimiter=',', usecols=range(0, 1), dtype=str).tolist()

            views.append(tuple(labels), (file_name, tuple(data)))
        return tuple(views)

    def mapper_init(self):
        views = self.read_data(self.views)
        for view in views:
            for i in range(len(view[0])):
                yield(view[0][i], (view[1][0], view[1][1][i]))

    def reducer(self, label, (view, features)):
        sample = [[], [], []]
        for v in view:
            for f in features:
                sample[v].append(features)
                
        for v in view:
            tuple(sample[v])
        tuple(sample)
        
        yield(label, sample)
        
if __name__ == '__main__':
    files = {0 : '100_leaves_plant_species/data_Mar_64.txt',
             1 : '100_leaves_plant_species/data_Sha_64.txt',
             2 : '100_leaves_plant_species/data_Tex_64.txt'}
    load_files = Load_Files(files)
    load_files.run()