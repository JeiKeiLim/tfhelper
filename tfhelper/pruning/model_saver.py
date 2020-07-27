import h5py


class ModelReducer:
    def __init__(self, model_path, out_path=None, method='gzip', opt=4):
        """
        Downsizing TensorFlow Keras Model File.

        :param model_path (str): Source model file path
        :param out_path (str, None): Destination model save path
        :param method (str): Reducing Method.
        :param opt (int): Reducing Optimization Level. (0 ~ 9). method=gzip only
        """
        self.in_path = model_path
        self.out_path = f"{model_path[:-3]}_reduced.h5" if out_path is None else out_path
        self.method = method
        self.opt = opt

    def reduce(self, debug=False):
        """
        Reducing TensorFlow Keras Model
        :param debug (bool): print debugging information and return reduced hd5f variable.
        :return (None, h5py.Dataset):
        """
        source = h5py.File(self.in_path, 'r')
        key_list = self.get_keys(source, root_list=[])

        with h5py.File(self.out_path, 'w') as f:
            for i, key in enumerate(key_list):

                if type(source[key]) == h5py._hl.dataset.Dataset:
                    if debug:
                        print(f"{i:02d} :: {key} - {source[key].dtype}, {source[key].shape}")
                    kwargs = {
                        "shape": source[key].shape,
                        "data": source[key],
                        "dtype": source[key].dtype,
                        "compression": self.method,
                    }
                    if self.method == "gzip":
                        kwargs["compression_opts"] = self.opt

                    if source[key].shape == ():
                        if kwargs.pop("compression") == "gzip":
                            kwargs.pop("compression_opts")

                    f.create_dataset(key, **kwargs)
                else:
                    if debug:
                        print(f"{i:02d} :: {key}")
                    f.create_group(key)

            keys = list(set(["/".join(key.split("/")[:i]) for key in key_list for i in range(0, len(key.split("/")))]))
            keys.remove("")

            for key in keys:
                for attr_key in source[key].attrs.keys():
                    f[key].attrs[attr_key] = source[key].attrs[attr_key]

            for attr_key in source.attrs.keys():
                f.attrs[attr_key] = source.attrs[attr_key]
        source.close()

        if debug:
            return h5py.File(self.out_path, 'r')

    def get_keys(self, source, root_list=None, root=''):
        """
        Recursive function to get all dict keys.
        :param source (h5py.Dataset, Dict): Source data.
        :param root_list (None, list):
        :param root (str): root path seperated by  /
        :return (list):

        """
        if root_list is None:
            root_list = []

        if type(source) == h5py._hl.dataset.Dataset or len(source) == 0:
            root_list.append(root)
            return root_list
        else:
            keys = source.keys()
            for key in keys:
                root_list = self.get_keys(source[key], root_list=root_list, root=f"{root}/{key}")

        return root_list
