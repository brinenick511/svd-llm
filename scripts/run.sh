MODEL_PATH=${HOME}/models/meta-llama/Llama-3.2-3B-Instruct
MODEL_PATH=${HOME}/models/meta-llama/Llama-3.1-8B-Instruct

GPU=3

python SVDLLM.py \
    --step 1  \
    --ratio 0.2 \
    --model ${MODEL_PATH} \
    --whitening_nsamples 256 \
    --dataset wikitext2 \
    --seed 3 \
    --model_seq_len 2048 \
    --save_path ${HOME}/svd-llm/cache/ \
    --DEV cuda:${GPU} \
    --decomposition v1 \
    --ratio_allocation None \
    --update None \
    # --pre o



