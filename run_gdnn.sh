export GRB_SPARSE_MATRIX_FORMAT=1
export GRB_UTIL_REMOVE_SELFLOOP=0
./bin/gdnn --directed 0 --debug 1 \
/home/wangxy/GraphChallenge/code/data/DNN/neuron1024-l120-categories.tsv \
/home/wangxy/GraphChallenge/code/data/MNIST/sparse-images-1024.mtx \
/home/wangxy/GraphChallenge/code/data/DNN/neuron1024/
