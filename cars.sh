python main.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
                --n_epochs 120 --project THSG --group CARS_THSG --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
                --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
                --split_subset_perc 0.25 --eval_frq 5 \
                --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
                --log_online --lr 1e-5 
                
python main.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
                --n_epochs 120 --project THSG --group CARS_THSG --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
                --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
                --split_subset_perc 0.25 --eval_frq 5 \
                --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
                --log_online --lr 2e-5 
                
python main.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
                --n_epochs 120 --project THSG --group CARS_THSG --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
                --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
                --split_subset_perc 0.25 --eval_frq 5 \
                --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
                --log_online --lr 5e-5 
                
python main.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
                --n_epochs 120 --project THSG --group CARS_THSG --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
                --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
                --split_subset_perc 0.25 --eval_frq 5 \
                --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
                --log_online --lr 1e-5 \
                --use_softmax --loss_softmax_lr 1e-5

python main.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
                --n_epochs 120 --project THSG --group CARS_THSG --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
                --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
                --split_subset_perc 0.25 --eval_frq 5 \
                --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
                --log_online --lr 1e-5 \
                --use_softmax --loss_softmax_lr 5e-5 --fc_lr 5e-5

python main.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
                --n_epochs 120 --project THSG --group CARS_THSG --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
                --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
                --split_subset_perc 0.25 --eval_frq 5 \
                --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
                --log_online --lr 1e-5 \
                --use_softmax --loss_softmax_lr 1e-4 --fc_lr 1.1e-4