# from https://github.com/osdf/datasets/blob/master/mnist/dataset.py
import h5py
import cPickle
import gzip
from os.path import dirname, join


_default_name = join(dirname(__file__), "mnist.h5")


def get_store(fname=_default_name):
    print("Loading from store {}".format(fname))
    return h5py.File(fname, 'r')


def build_store(store=_default_name, mnist="mnist.pkl.gz"):
    """Build a hdf5 data store for MNIST.
    """
    print("Reading {}").format(mnist)
    mnist_f = gzip.open(mnist,'rb')
    train_set, valid_set, test_set = cPickle.load(mnist_f)
    mnist_f.close()

    print("Writing to {}").format(store)
    h5file = h5py.File(store, "w")

    print("Creating train set.")
    grp = h5file.create_group("train")
    dset = grp.create_dataset("inputs", data = train_set[0])
    dset = grp.create_dataset("targets", data = train_set[1])

    print("Creating validation set.")
    grp = h5file.create_group("validation")
    dset = grp.create_dataset("inputs", data = valid_set[0])
    dset = grp.create_dataset("targets", data = valid_set[1])

    print("Creating test set.")
    grp = h5file.create_group("test")
    dset = grp.create_dataset("inputs", data = test_set[0])
    dset = grp.create_dataset("targets", data = test_set[1])

    print("Closing {}").format(store)
    h5file.close()


if __name__=="__main__":
    build_store()