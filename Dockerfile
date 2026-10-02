# Build:
#   docker build -t nngp-project .
#
# Run (reproduces both Figure 3 panels + sigmoid extension):
#   docker run nngp-project
#
# Run with the default (MNIST) command, mounting a folder so the PNG
# comes back out onto your machine:
#   mkdir -p output
#   docker run --platform linux/amd64 -v "$(pwd)/output":/nngp/output nngp-project
#
# Override the default to run CIFAR-10 instead (or change any flag):
#   docker run --platform linux/amd64 -v "$(pwd)/output":/nngp/output nngp-project \
#       --dataset=cifar10 --num_train=1000 --num_eval=1000 \
#       --hparams='depth=3,weight_var=2.0,bias_var=0.2' \
#       --nonlinearities='tanh,relu,sigmoid' \
#       --output_file=/nngp/output/uncertainty_fig3_cifar.png

# The base image is amd64-only; pinning the platform makes a plain
# `docker build` also work on Apple Silicon (via emulation).
FROM --platform=linux/amd64 tensorflow/tensorflow:1.15.0-py3

WORKDIR /nngp

# matplotlib isn't in the base TensorFlow image, and is needed for the plot
RUN pip install --no-cache-dir matplotlib

# Pre-download both datasets into the image so `docker run` never downloads.
# Paths are the ones the code already reads from: load_dataset.py's default
# --data_dir (/tmp/nngp/data/) and Keras's cache (~/.keras/datasets).
# Placed before COPY so editing repo files doesn't trigger a re-download.
RUN python -c "from tensorflow.examples.tutorials.mnist import input_data; input_data.read_data_sets('/tmp/nngp/data/', False, validation_size=10000, one_hot=True)" && \
    python -c "from tensorflow.keras.datasets import cifar10; cifar10.load_data()"

# Copy everything in the repo (original nngp source, grid_data/, and
# uncertainty_plot.py) into the image.
COPY . /nngp

# Two fixes needed for this old codebase to run on a current image, baked
# in at build time instead of by hand each container session:
#   1. Python 2 -> 3 (xrange doesn't exist in Python 3)
#   2. numpy now defaults to allow_pickle=False; the precomputed grid files
#      in grid_data/ were saved as pickled object arrays
RUN sed -i 's/\bxrange\b/range/g' *.py && \
    sed -i "s/np.load(f)/np.load(f, allow_pickle=True, encoding='latin1')/" \
        nngp.py && \
    sed -i 's/parallel_iterations=multiprocessing.cpu_count()/parallel_iterations=1/' \
        nngp.py


# Sigmoid extension: the repo only ships tanh/relu grids, so precompute the
# sigmoid lookup table (paper Section 2.5, Eq. 10) once at build time, with
# the same resolution flags uncertainty_plot.py uses. Takes ~10-20 min. The
# `ls` fails the build if the file isn't named what the script looks for.
RUN python -c "import tensorflow as tf, nngp; tf.app.flags.FLAGS(['nngp']); nngp.NNGPKernel(nonlin_fn=tf.nn.sigmoid, grid_path='./grid_data', n_gauss=501, n_var=501, n_corr=500, max_gauss=10, max_var=100)" && \
    ls -l grid_data/grid_sigmoid_ng501_ns501_nc500_mv100_mg10

# `docker run nngp-project` with no extra args reproduces Figure 3: one
# figure per dataset (the paper's two panels), each with Tanh, ReLU, and the
# sigmoid extension, at depth=3, weight_var=2.0, bias_var=0.2 (Figure 3
# caption). Both runs must succeed for the container to exit cleanly.
CMD ["sh", "-c", "\
python uncertainty_plot.py --dataset=mnist --num_train=1000 --num_eval=5000 \
  --hparams=depth=3,weight_var=2.0,bias_var=0.2 \
  --nonlinearities=tanh,relu,sigmoid \
  --output_file=/nngp/output/uncertainty_fig3_mnist.png && \
python uncertainty_plot.py --dataset=cifar10 --num_train=1000 --num_eval=5000 \
  --hparams=depth=3,weight_var=2.0,bias_var=0.2 \
  --nonlinearities=tanh,relu,sigmoid \
  --output_file=/nngp/output/uncertainty_fig3_cifar.png"]
