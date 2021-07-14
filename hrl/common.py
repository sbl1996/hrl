from collections import namedtuple

Transition = namedtuple('Transition',
                        ('obs', 'action', 'obs_next', 'reward', 'done'))
