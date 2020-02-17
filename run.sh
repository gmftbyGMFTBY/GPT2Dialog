mode=$1
cuda=$2

if [ $mode = 'train' ]; then
    CUDA_VISIBLE_DEVICES="$2" python train.py --seed 30 --epochs 100 --batch_size 16 --device 0
elif [ $mode = 'test' ]; then
    CUDA_VISIBLE_DEVICES="$2" python interact.py
fi
