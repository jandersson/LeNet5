import pathlib
import struct
import gzip
import numpy as np
from torch.utils.data import Dataset

class mnist(Dataset):
    """Data and data format specification at http://yann.lecun.com/exdb/mnist/"""
    def __init__(self, set_type='train', transform=None):
        if set_type not in ['train', 't10k', 'test']:
            raise Exception('Unrecognized data set choice. Valid choices are "train", "test", or "t10k"')
        if set_type == 'test':
            set_type = 't10k'
        self.set_type = set_type
        self.transform = transform
        self.stdev = None
        self.mean = None
        self._load_data()
        self._set_stdev()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx][0], 'label': self.data[idx][1]}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _load_data(self):
        images, labels = [[], []]
        set_type = self.set_type
        data_path = pathlib.Path(__file__).resolve().parent / 'data' / 'mnist'
        for compressed_file in data_path.glob('*.gz'):
            if set_type not in compressed_file.name:
                continue
            with gzip.open(compressed_file, 'rb') as cf:
                if 'labels' in compressed_file.name:
                    # Unpack magic number (discarded) and number of elements
                    _magic_number, num_labels = struct.unpack('>ii', cf.read(8))
                    # Unpack the rest into a list
                    labels_iter = struct.iter_unpack('>B', cf.read())
                    labels = np.array([label[0] for label in labels_iter])
                    assert num_labels == len(labels)
                elif 'images' in compressed_file.name:
                    # Unpack the magic number (discarded), number of images, number of rows, number of columns
                    images = []
                    _magic_number, num_images, self.num_rows, self.num_cols = struct.unpack('>iiii', cf.read(16))
                    pixels = list(struct.iter_unpack('>B', cf.read()))
                    pix_sum = 0.0
                    num_pixels = 0
                    for i in range(0, num_images * self.num_rows * self.num_cols, self.num_rows * self.num_cols):
                        image = np.array([pixel[0] for pixel in pixels[i: i + self.num_rows * self.num_cols]], dtype=float)
                        pix_sum += np.sum(image)
                        num_pixels += len(image)
                        image.shape = (self.num_rows, self.num_cols, 1)
                        images.append(image)
                    self.pix_mean = pix_sum/num_pixels
                    self.num_pixels = num_pixels
                    assert len(images) == num_images
        assert len(images) == len(labels)
        self.data = list(zip(images, labels))

    def _set_stdev(self):
        variance_sum = 0.0
        for image, _ in self.data:
            for pix in image.ravel():
                variance_sum += (pix - self.pix_mean)**2
        avg_variance = variance_sum/self.num_pixels
        self.stdev = np.sqrt(avg_variance)
