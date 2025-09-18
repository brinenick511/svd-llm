META=1

MODEL_PATH=/data/yanghq/test_swift/llama-wtf-1${META}

python SVDLLM.py \
    --step 1  \
    --ratio 0.2 \
    --model ${MODEL_PATH} \
    --whitening_nsamples 256 \
    --dataset wikitext2 \
    --seed 3 \
    --model_seq_len 2048 \
    --save_path ${HOME}/svd-llm/cache/ \
    --DEV cuda:${META} \
    --decomposition v1 \
    --ratio_allocation None \
    --update None \
    --pre None

python SVDLLM.py \
    --step 4 \
    --model_path original \
    --model ${MODEL_PATH} \
    --eval_batch_size 2 \
    --dataset wikitext2 \
    --seed 3 \
    --model_seq_len 2048 \
    --DEV cuda:${META}

