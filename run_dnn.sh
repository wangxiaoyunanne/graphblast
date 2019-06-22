# Mario
datadir=/data-3/GraphChallenge19-DNN/data

# # Luigi
# datadir=/data-2/GraphChallenge19

export GRB_SPARSE_MATRIX_FORMAT=1
export GRB_UTIL_REMOVE_SELFLOOP=0
export GRB_MXM_CUSPARSE_MODE=2

./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --endbit 0 --transpose 1 --nlayer 120 --nneuron 65536 --batch_size 5000 ${datadir}
./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --endbit 0 --transpose 1 --nlayer 480 --nneuron 65536 --batch_size 5000 ${datadir}

for NN in 16384 65536
#for NN in 1024 4096 16384 65536
do
  for NL in 480 1920
  #for NL in 120 480 1920
  do
    echo $NN $NL
    echo ./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --endbit 0 --transpose 1 --nlayer $NL --nneuron $NN --batch_size 15000 ${datadir}
    #./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --endbit 0 --transpose 1 --nlayer $NL --nneuron $NN --batch_size 15000 ${datadir}
  done
done
