set -x
set -e

export CUDA_VISIBLE_DEVICES=$1

python detector_yang.py \
            --checkpoint_path=$2 \
            --dataset_dir=$3 \
            --eval_image_width=800\
            --eval_image_height=400\
            --pixel_conf_threshold=0.7\
            --link_conf_threshold=0.7\
            --gpu_memory_fraction=-1
