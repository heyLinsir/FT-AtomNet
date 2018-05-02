from tensorflow.examples.tutorials.mnist import input_data

class Data(object):
    """docstring for Data."""
    def __init__(self, config):
        super(Data, self).__init__()
        self.config = config
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    def get_train_data(self, batch_size=-1):
        if batch_size < 0:
            batch_size = self.config.batch_size
        batch_cnt = int(len(self.mnist.train.images) / batch_size)
        for i in range(batch_cnt):
            feed_dict = {}
            batch = self.mnist.train.next_batch(batch_size)
            feed_dict['x'] = batch[0]
            feed_dict['label'] = batch[1]
            yield feed_dict

    def get_test_data(self, batch_size=-1):
        if batch_size < 0:
            batch_size = self.config.batch_size
        batch_cnt = int(len(self.mnist.test.images) / batch_size)
        for i in range(batch_cnt):
            feed_dict = {'x': self.mnist.test.images[i * batch_size:(i + 1) * batch_size], \
                        'label': self.mnist.test.labels[i * batch_size:(i + 1) * batch_size]}
            yield feed_dict

    def get_dev_data(self, batch_size=-1):
        if batch_size < 0:
            batch_size = self.config.batch_size
        batch_cnt = int(len(self.mnist.validation.images) / batch_size)
        for i in range(batch_cnt):
            feed_dict = {'x': self.mnist.validation.images[i * batch_size:(i + 1) * batch_size], \
                        'label': self.mnist.validation.labels[i * batch_size:(i + 1) * batch_size]}
            yield feed_dict

if __name__ == '__main__':
    data = Data(None)
