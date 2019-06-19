# # Mario
# datadir=/home/wangxy/GraphChallenge/code/data

# Luigi
datadir=/data-2/GraphChallenge19

export GRB_SPARSE_MATRIX_FORMAT=1
export GRB_UTIL_REMOVE_SELFLOOP=0
./bin/gdnn --directed 0 --debug 0 \
${datadir}/DNN/neuron1024-l120-categories.tsv \
${datadir}/MNIST/sparse-images-1024.mtx \
${datadir}/DNN/neuron1024/
