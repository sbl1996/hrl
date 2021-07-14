from tensorflow.keras import Sequential
from hanser.models.layers import Linear

class MLP(Sequential):

    def __init__(self, in_channels, out_channels, hidden_sizes, act='relu'):
        self.in_channels = in_channels
        channels = (in_channels,) + tuple(hidden_sizes)
        layers = [
            Linear(i, o, act=act, bias_init='zeros')
            for i, o in zip(channels[:-1], channels[1:])
        ]
        layers.append(Linear(channels[-1], out_channels, bias_init='zeros'))
        super().__init__(layers)
