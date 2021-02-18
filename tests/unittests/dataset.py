# import sys
# import time
# from functools import partial
# import logging
# import numpy as onp
# import jax
# import jax.numpy as jnp
# from deepexcite.dataset import ConcatSequence, DataStream, imresize

# # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# def test_getitem():
#     root = r"/media/ep119/DATADRIVE3/epignatelli/deepexcite/train_dev_set/"
#     search_regex = r"^spiral.*_PARAMS5.hdf5"
#     batch_size = 32
#     frames_in = 2
#     frames_out = 1
#     step = 5
#     n_channels = 5
#     size = (96, 96)
#     shape = (frames_in + frames_out, 5, *size)
#     preload = True
#     seed = 0

#     dataset = ConcatSequence(
#         root=root,
#         frames_in=frames_in,
#         frames_out=frames_out,
#         step=step,
#         transform=partial(imresize, size=size, method="bilinear"),
#         keys=search_regex,
#         preload=preload,
#     )

#     loader = DataStream(dataset, batch_size, jnp.stack, seed)
#     for i in range(1, 20):
#         loader.frames_out = i
#         logging.info("Frames out is {}".format(i))
#         start = time.time()
#         for j, batch in enumerate(loader):
#             assert batch.shape == (batch_size, i + frames_in, 5, *size), \
#                 "Batch size is incorrect at index {} for refeed {}. Should be {} but is {}".format(
#                     j,
#                     i,
#                     (batch_size, i + frames_in, 5, *size),
#                     batch.shape
#                 )
#         print("elapsed: {}".format(time.time() - start))

#     return True
