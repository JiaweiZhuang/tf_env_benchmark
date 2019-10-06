import argparse
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument('--num_threads', type=int, default=1, help='number of CPU threads')
parser.add_argument('--repeat', type=int, default=5, help='number of repeated runs')

parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--grid_size', type=int, default=256, help='input grid size')

parser.add_argument('--num_inputs', type=int, default=3, help='number of input features')
parser.add_argument('--num_outputs', type=int, default=3, help='number of output features')
parser.add_argument('--num_layers', type=int, default=4, help='number of conv2d layers')
parser.add_argument('--filters', type=int, default=32, help='number of filters in each layer')
parser.add_argument('--kernel_size', type=int, default=3, help='conv2d kernel size')

args = parser.parse_args()


def standard_conv2d_stack(num_outputs, num_layers=5, filters=32, kernel_size=5,
                          activation='relu', **kwargs):
    """Create a sequence of Conv2D layers."""
    model = tf.keras.Sequential()
    for _ in range(num_layers - 1):
        layer = tf.keras.layers.Conv2D(
            filters, kernel_size, activation=activation, padding='same', **kwargs)
        model.add(layer)
    model.add(tf.keras.layers.Conv2D(num_outputs, kernel_size, padding='same', **kwargs))
    return model


if __name__ == '__main__':
    num_threads = args.num_threads
    repeat = args.repeat
    
    if tf.__version__ < '2.0.0':
        config = tf.ConfigProto(
            intra_op_parallelism_threads=num_threads,
            inter_op_parallelism_threads=num_threads
        )
        tf.enable_eager_execution(config=config)
    else:
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        
    
    model = standard_conv2d_stack(
        args.num_outputs, 
        num_layers=args.num_layers, 
        filters=args.filters, 
        kernel_size=args.kernel_size
        )
    inputs = tf.zeros([args.batch_size, args.grid_size, args.grid_size, args.num_inputs])


    start = timer()
    for _ in range(repeat):
        output = model(inputs)
    end = timer()
    
    elapsed = (end - start) / repeat
    print('average time: {:.4f} seconds'.format(elapsed))
