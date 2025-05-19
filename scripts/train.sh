LOSS_TYPE=$1 # e.g., contrasive, ce
SPURIOUS=$2 # e.g., spurious_bg, spurious_texture
SPURIOUS_TYPE=$3 # e.g., star, kl
REG_RATIO=$4  

KEEP_LANG=true

SEED=("0")
export PYTHONPATH="$PYTHONPATH:$PWD"
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -k|--keep-lang)
            KEEP_LANG=true
            shift # past argument
            ;;
        *)  
    esac
    shift # past argument or value
done

for seed in "${SEED[@]}"; do
    python src/main.py \
        --train-dataset "ImageNet" \
        --epochs 10 \
        --lr 1e-5 \
        --wd 0.1 \
        --batch-size 512 \
        --model "ViT-B/16" \
        --eval-datasets ImageNet,ImageNetV2,ImageNetR,ImageNetA,ImageNetSketch \
        --template "openai_imagenet_template" \
        --save "./logs/" \
        --data-location "./datasets/" \
        --ft-data "keywords/imagenet/imagenet_base.csv" \
        --spurious-path "keywords/${SPURIOUS}.pt" \
        --spurious-type "${SPURIOUS_TYPE}" \
        --exp-name "flyp" \
        --reg-ratio "${REG_RATIO}" \
        --loss-type "${LOSS_TYPE}" \
        $(if [ "${KEEP_LANG}" = true ]; then echo "--keep-lang"; fi) \
        --diminishing
        
#
done
