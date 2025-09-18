MODEL_PATH=${HOME}/models/meta-llama/Llama-3.2-1B-Instruct
MODEL_PATH=/data/yanghq/svd-llm/cache/_data_yanghq_models_meta_llama_Llama_3.2_1B_Instruct_whitening_only_0.8.pt
MODEL_PATH=${HOME}/models/meta-llama/Llama-3.1-8B-Instruct
MODEL_PATH=/data/yanghq/svd-llm/cache/_data_yanghq_models_meta_llama_Llama_3.1_8B_Instruct_whitening_only_0.8.pt
MODEL_PATH=/data/yanghq/test_swift/Llama-3.2-3B-Instruct-merged
MODEL_PATH=/data/yanghq/svd-llm/cache/_data_yanghq_test_swift_Llama_3.2_3B_Instruct_merged_whitening_only_0.8.pt
MODEL_PATH=/data/yanghq/test_swift/llama-wtf-32
MODEL_PATH=/data/yanghq/svd-llm/cache/_data_yanghq_test_swift_llama_wtf_63_whitening_only_0.8.pt
MODEL_PATH=/data/yanghq/svd-llm/cache/_data_yanghq_test_swift_llama_wtf_11_whitening_only_0.8.pt
BATCH_SIZE=2
GPU=1



python SVDLLM.py \
    --step 4 \
    --model_path ${MODEL_PATH} \
    --eval_batch_size ${BATCH_SIZE} \
    --dataset wikitext2 \
    --seed 3 \
    --model_seq_len 2048 \
    --DEV cuda:${GPU}


# python SVDLLM.py \
#     --step 4 \
#     --model_path original \
#     --model ${MODEL_PATH} \
#     --eval_batch_size ${BATCH_SIZE} \
#     --dataset wikitext2 \
#     --seed 3 \
#     --model_seq_len 2048 \
#     --DEV cuda:${GPU}

