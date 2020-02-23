# run.sh <train> <0> <dailydialog>
mode=$1
cuda=$2
dataset=$3

if [ $mode = 'train' ]; then
    CUDA_VISIBLE_DEVICES="$2" python train.py \
        --raw \
        --seed 30 \
        --epochs 200 \
        --batch_size 16 \
        --device 0 \
        --train_raw_path data/$3/train.txt \
        --train_tokenized_path data/$3/train_tokenized.txt \
        --log_path data/$3/training.log \
        --dialogue_model_output_path dialogue_model/$3/ \

        --
elif [ $mode = 'test' ]; then
    CUDA_VISIBLE_DEVICES="$2" python interact.py \
        --dialogue_model_path dialogue_model/$3/model_epoch16 \
        --test_data_path data/$3/src-test.txt \
        --save_samples_path data/$3/pred.txt \

fi
