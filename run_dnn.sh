# Mario
datadir=/data-3/GraphChallenge19-DNN/data

# # Luigi
# datadir=/data-2/GraphChallenge19

export GRB_SPARSE_MATRIX_FORMAT=1
export GRB_UTIL_REMOVE_SELFLOOP=0
export GRB_MXM_CUSPARSE_MODE=2

for NN in 1024 4096
do
  for NL in 120 480 1920
  do
    echo $NN $NL
    echo ./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --transpose 1 --nlayer $NL --nneuron $NN ${datadir}
    ./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --transpose 1 --nlayer $NL --nneuron $NN ${datadir}
  done
done

echo 16384 120
./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --transpose 1 --nlayer 120 --nneuron 16384 --batch_size 30000 ${datadir}
echo 16384 480
./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --transpose 1 --nlayer 480 --nneuron 16384 --batch_size 30000 ${datadir}
echo 16384 1920
./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --transpose 1 --nlayer 1920 --nneuron 16384 --batch_size 5000 ${datadir}
echo 65536 120
./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --transpose 1 --nlayer 120 --nneuron 65536 --batch_size 5000 ${datadir}
echo 65536 480
./bin/gdnn --mtxinfo 0 --directed 0 --debug 0 --filter 1 --transpose 1 --nlayer 480 --nneuron 65536 --batch_size 1000 ${datadir}
