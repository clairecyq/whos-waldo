STORAGE=$1 # Stores processed data for the model and the finetuning results
DATASET_META=$2 # Metadata about the dataset
WHOS_WALDO=$3 # Who's Waldo dataset

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$STORAGE,dst=/storage,type=bind \
    --mount src=$DATASET_META,dst=/dataset_meta,type=bind \
    --mount src=$WHOS_WALDO,dst=/whos_waldo,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /src claire43/uniter
