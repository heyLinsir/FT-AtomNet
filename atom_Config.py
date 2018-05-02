class Config(object):
    """docstring for Config"""
    def __init__(self):
        super(Config, self).__init__()
        self.sgd_learning_rate = 1e-1
        self.adam_learning_rate = 1e-4
        self.optimizer = 'Adam' # Adam or Adagrad or Adadelta or SGD

        self.num_class = 10
        self.keep_prob = 0.5
        self.num_sample = 30000
        self.input_dim = 28 * 28

        self.batch_size = 128
        self.epoch = 100
        self.checkpoints = './checkpoints_first_version'
        self.use_latest_params = False
