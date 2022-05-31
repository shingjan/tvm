set -euxo pipefail

RPC_HOST=172.31.53.201
RPC_PORT=4445
LOG_ID=1

RPC_KEY="c5.9xlarge"
TARGET="llvm -mcpu=skylake-avx512 -num-cores 18"
#TARGET="llvm -num-cores 18"
LOG_DIR=$HOME/tvm/logs/perf/llvm/yolov4/

RPC_WORKERS=1
CMD="tvm.meta_schedule.testing.tune_relay_meta_schedule"
today=$(date '+%F_%T')
tuner="ms"

mkdir -p $LOG_DIR

run () {
  workload="$1"
  input_shape="$2"
  num_trials="$3"
  WORK_DIR=$LOG_DIR/${tuner}_${workload}_${today}/
  mkdir -p $WORK_DIR
  rm -f $WORK_DIR/*.json

  python3 -m $CMD                      \
    --workload "$workload"            \
    --input-shape "$input_shape"      \
    --model-type "onnx"               \
    --target "$TARGET"                \
    --num-trials $num_trials          \
    --rpc-host $RPC_HOST              \
    --rpc-port $RPC_PORT              \
    --rpc-key $RPC_KEY                \
    --rpc-workers $RPC_WORKERS        \
    --work-dir $WORK_DIR              \
    --cache-dir $HOME/cache-workloads/ \
    |& tee "$WORK_DIR/${workload}_${today}.log"
}

run "yolov4"     "[1,64]"          20000
