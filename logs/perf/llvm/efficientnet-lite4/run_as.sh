set -euxo pipefail

RPC_HOST=172.31.53.201
RPC_PORT=4445
LOG_ID=1

RPC_KEY="c5.9xlarge"
TARGET="llvm -mcpu=skylake-avx512 -num-cores 18"
LOG_DIR=$HOME/tvm/logs/perf/llvm/efficientnet-lite4/

RPC_WORKERS=4
CMD="tvm.meta_schedule.testing.tune_relay_auto_scheduler"
today=$(date '+%F_%T')
tuner="as"

mkdir -p $LOG_DIR

run () {
  workload="$1"
  input_shape="$2"
  num_trials="$3"

  python -m $CMD                      \
    --workload "$workload"            \
    --input-shape "$input_shape"      \
    --model-type "onnx"               \
    --target "$TARGET"                \
    --num-trials $num_trials          \
    --rpc-host $RPC_HOST              \
    --rpc-port $RPC_PORT              \
    --rpc-key $RPC_KEY                \
    --rpc-workers $RPC_WORKERS        \
    --log-dir $LOG_DIR                \
    --cache-dir $HOME/cache-workloads/ \
    2>&1 | tee -a "$LOG_DIR/${tuner}_${workload}_${today}.log"
}

run "efficientnet-lite4"     "[1,64]"          20000

