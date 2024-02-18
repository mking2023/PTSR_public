
seed_list=(2023) 

python_script="main.py"

for seed in "${seed_list[@]}"; do
    
    printf "#%.0s" {1..80}
    echo
    echo "Executing $python_script with seed $seed"

    python "$python_script" \
        --epoch 200 \
        --eval_interval 5 \
        --lr 0.001 \
        --batchsize 512 \
        --batchsize_eval 512 \
        --iter_interval 5 \
        --seed $seed \
        --gpu 1 \
        --max_len 20 \
        --emb_dim 64 \
        --base_add 1 \
        --gamma 2 \
        --pattern_level 2 \
        --dataset "Beauty" \
        --emb_type "gamma" \
        --mlp_lambda 0.4 \
        --weight_decay 0.00000001 
done
