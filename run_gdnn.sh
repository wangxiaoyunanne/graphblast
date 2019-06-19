# Mario
datadir=/data-3/GraphChallenge19-DNN/data

# # Luigi
# datadir=/data-2/GraphChallenge19

export GRB_SPARSE_MATRIX_FORMAT=1
export GRB_UTIL_REMOVE_SELFLOOP=0
./bin/gdnn --directed 0 --debug 0 ${datadir}
