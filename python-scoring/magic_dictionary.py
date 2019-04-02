from __future__ import print_function
from builtins import str, range, object
from abc import ABCMeta, abstractmethod  # enables abstract base classes
from future.utils import with_metaclass
import numpy as np
import os
import gzip
from tempfile import mkdtemp
from timeit import default_timer as timer


# factory for appropriate type of dictionary. Assumes we want to compute all pairs and store them in a square ndarray.
def make_me_a_dict(num_indiv_items, data_dir = None, force_memmap = False):
    if num_indiv_items > 3000 or force_memmap:    # 3000 items --> ~10 M entries in matrix
        return(onDiskDict(data_dir, (num_indiv_items, num_indiv_items)))
    else:
        return(normalInMemory((num_indiv_items, num_indiv_items)))


class MagicDictionary(with_metaclass(ABCMeta, object)):

    def __init__(self, ndarray_shape):
        self.underlying_dict = {}
        self.ndarray_shape = ndarray_shape
        self.hidden_items = set()

    @abstractmethod
    # use "int" and "float" to have printing handled right
    def create_and_store_array(self, key, dtype):
        pass

    # works just like the usual create_and_store_array(), except these are added to a special list
    # of variables that don't show up as columns in the outfile
    def create_and_store_unofficial(self, key, dtype):
        self.hidden_items.add(key)
        return self.create_and_store_array(key, dtype)

    @abstractmethod
    def retrieve_array(self, key):
        pass

    @abstractmethod
    def retrieve_all_arrays(self, methods):
        pass

    def getkeys(self):
        return(set(self.underlying_dict.keys()) - self.hidden_items)

    def rename_method(self, old_name, new_name):
        self.underlying_dict[new_name] = self.underlying_dict.pop(old_name)


    def to_csv_gz(self, outfile, pairs_generator, pg_arg):
        methods = sorted(self.getkeys())
        # get all arrays into 1 place (now, a dictionary)
        vals = self.retrieve_all_arrays(methods)

        header = ",".join(['item1', 'item2'] + methods)
        with gzip.open(outfile, 'wb') as fout:
            fout.write(header + "\n")
            for (i, j, _, _, _, _) in pairs_generator(pg_arg):
                fout.write(",".join([str(i), str(j)] + [str(vals[m][i,j]) for m in methods]) + "\n")


# Stores a dictionary of ndarrays
class normalInMemory(MagicDictionary):
    def __init__(self, ndarray_shape):
        super(normalInMemory, self).__init__(ndarray_shape)

    def create_and_store_array(self, key, dtype=float):
        self.underlying_dict[key] = np.zeros(self.ndarray_shape, dtype=dtype)
        return(self.underlying_dict[key])

    def retrieve_array(self, key):
        return(self.underlying_dict[key])

    def retrieve_all_arrays(self, methods):
        # arrays = [self.retrieve_array(m) for m in methods]
        # return(np.stack(arrays, axis=0))
        return(self.underlying_dict)



# Internal dictionary stores filenames of the memmap .dat files
class onDiskDict(MagicDictionary):
    def __init__(self, data_dir, ndarray_shape):
        super(onDiskDict, self).__init__(ndarray_shape)
        self.data_dir_is_temp = False
        if data_dir is None:
            data_dir = mkdtemp()
            self.data_dir_is_temp = True
        else:
            if not os.path.isdir(data_dir):
                os.mkdir(data_dir)      # throws error if unhappy
        self.data_dir = data_dir
        print("storing temp .dat files in " + data_dir)
        self.types = {}

    def get_filename_for_key(self, key):
        return(os.path.join(self.data_dir, key + ".dat"))

    def create_and_store_array(self, key, dtype=float):
        filename = self.get_filename_for_key(key)
        data = np.memmap(filename, dtype=dtype, mode="w+", shape=self.ndarray_shape)
        self.underlying_dict[key] = filename
        self.types[key] = dtype  # must store, in order to open correctly later
        return(data)

    def retrieve_array(self, key):  # read-only, unless there's a reason to change it
        return(np.memmap(self.underlying_dict[key], mode="r", shape=self.ndarray_shape, dtype=self.types[key]))

    def retrieve_all_arrays(self, methods):
        # self.retrieve_all = os.path.join(self.data_dir, "_retrieve_all" + ".dat")
        # out_array = np.memmap(self.retrieve_all, dtype=float, mode="w+", shape=[len(methods)] + self.ndarray_shape)
        # arrays = [self.retrieve_array(m) for m in methods]
        # return(np.stack(arrays, axis=0, out=out_array))
        return({m:self.retrieve_array(m) for m in methods})

    # should be better for conserving RAM
    def to_csv_gz(self, outfile, pairs_generator, pg_arg):
        methods = sorted(self.getkeys())
        if len(methods) <= 1:
            super(onDiskDict, self).to_csv_gz(outfile, pairs_generator, pg_arg)

        # for each method, re-save only the scores we want, to a 1-col text file
        for m in methods:
            new_file = os.path.join(self.data_dir, m + ".2.dat")
            data = self.retrieve_array(m)
            with open(new_file, 'w') as fout:
                # fout.write(m + "\n")
                for (i,j,_,_,_,_) in pairs_generator(pg_arg):
                    fout.write(str(data[i,j]) + "\n")

        # join into a csv file
        with gzip.open(outfile, 'wb') as fout:
            infps = []
            for m in methods:
                new_file = os.path.join(self.data_dir, m + ".2.dat")
                infps.append(open(new_file, 'r'))
            fout.write("item1,item2," + ",".join(methods) + "\n")
            for (i, j, _, _, _, _) in pairs_generator(pg_arg):
                fout.write(",".join([str(i), str(j)] + [f.readline().rstrip() for f in infps]) + "\n")

            for fp in infps:
                fp.close()

        for m in methods:
            os.remove(os.path.join(self.data_dir, m + ".2.dat"))


    def __del__(self):
        # clean up, removing the temp files
        for filename in self.underlying_dict.values():
            os.remove(filename)
        if self.data_dir_is_temp:
            os.rmdir(self.data_dir)

    def rename_method(self, old_name, new_name):
        self.types[new_name] = self.types.pop(old_name)
        os.rename(self.get_filename_for_key(old_name), self.get_filename_for_key(new_name))
        self.underlying_dict[new_name] = self.get_filename_for_key(new_name)
        del self.underlying_dict[old_name]
