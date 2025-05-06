#!/bin/bash
##############################################################################
# NATS-Bench: Benchmarking NAS algorithms for Architecture Topology and Size #
##############################################################################
echo script name: $0
echo $# arguments

if [ "$#" -ne 4 ]; then
  echo "Illegal number of parameters: $#. Expected 4."
  echo "Usage: bash scripts/NATS-Bench/train-topology.sh <start-end> <hyper-params-opt-file> <seeds> <dataset>"
  echo "Example: bash scripts/NATS-Bench/train-topology.sh 00000-02000 200 \"777 888 999\" synthetic-cifar10"
  exit 1
fi

if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME environment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

srange=$1
opt=$2
all_seeds=$3
dataset=$4
cpus=4
save_dir=./output/NATS-Bench-topology/

# Define dataset-specific options
case $dataset in
  synthetic-cifar10)
    dataset_arg="synthetic-cifar10"
    xpaths_arg="$TORCH_HOME/synthetic"
    splits_arg="0"
    ;;
  synthetic-tiny-imagenet10 | synthetic-Tiny10 | synthetic-tiny10)
    dataset_arg="synthetic-Tiny10"
    xpaths_arg="$TORCH_HOME/synthetic"
    splits_arg="0"
    ;;
  cifar10 | cifar100 | ImageNet16-120)
    dataset_arg="cifar10 cifar10 cifar100 ImageNet16-120"
    xpaths_arg="$TORCH_HOME/cifar.python $TORCH_HOME/cifar.python $TORCH_HOME/cifar.python $TORCH_HOME/cifar.python/ImageNet16"
    splits_arg="1 0 0 0"
    ;;
  Tiny10)
    dataset_arg="Tiny10"
	xpaths_arg="$TORCH_HOME/tiny10"
	splits_arg="0"
	;;
  *)
    echo "Unknown dataset: $dataset"
    exit 1
    ;;
esac

# Run Python script with appropriate arguments
OMP_NUM_THREADS=${cpus} python exps/NATS-Bench/main-tss.py \
  --mode new \
  --srange ${srange} \
  --hyper ${opt} \
  --save_dir ${save_dir} \
  --datasets ${dataset_arg} \
  --splits ${splits_arg} \
  --xpaths ${xpaths_arg} \
  --workers ${cpus} \
  --seeds ${all_seeds}
