# python main_hdml.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
#                 --n_epochs 120 --project THSG --group CARS_THSG_S1 --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
#                 --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
#                 --split_subset_perc 0.25 --eval_frq 5 \
#                 --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
#                 --log_online --lr 1.5e-5 --generation \
#                 --use_softmax --loss_softmax_lr 1.5e-5 --fc_lr 1.6e-5 \
#                 --gen_lr 1e-4 --dis_lr 1e-4
                
# # random hdml
# python main_hdml.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
#                 --n_epochs 120 --project THSG --group CARS_THSG_S1 --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
#                 --loss triplet --batch_mining random --arch resnet50_frozen_normalize \
#                 --split_subset_perc 0.25 --eval_frq 5 \
#                 --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
#                 --log_online --lr 1.5e-5 --generation \
#                 --use_softmax --loss_softmax_lr 1.5e-5 --fc_lr 2e-5 \
#                 --gen_lr 5e-4 --dis_lr 5e-4

# python main_hdml.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
#                 --n_epochs 120 --project THSG --group CARS_THSG_S1_D1 --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
#                 --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
#                 --split_subset_perc 0.25 --eval_frq 5 \
#                 --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
#                 --log_online --lr 1.5e-5 --generation \
#                 --use_softmax --loss_softmax_lr 1.5e-5 --fc_lr 2e-5 \
#                 --gen_lr 1e-4 --dis_lr 1e-4

# # best hdml & D1
# python main_hdml.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
#                 --n_epochs 120 --project THSG --group CARS_THSG_S1_D1 --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
#                 --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
#                 --split_subset_perc 0.25 --eval_frq 5 \
#                 --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
#                 --log_online --lr 1.5e-5 --generation \
#                 --use_softmax --loss_softmax_lr 1.5e-5 --fc_lr 2e-5 \
#                 --gen_lr 5e-4 --dis_lr 5e-4

# python main_hdml.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
#                 --n_epochs 120 --project THSG --group CARS_THSG_S1_D1 --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
#                 --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
#                 --split_subset_perc 0.25 --eval_frq 5 \
#                 --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
#                 --log_online --lr 1.5e-5 --generation \
#                 --use_softmax --loss_softmax_lr 1.5e-5 --fc_lr 2e-5 \
#                 --gen_lr 1e-3 --dis_lr 1e-3

# # hdml & D1 & J_fan
# python main_s1_rev.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
#                 --n_epochs 120 --project THSG --group CARS_THSG_S1_D1_Jfan --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
#                 --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
#                 --split_subset_perc 0.25 --eval_frq 5 \
#                 --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
#                 --log_online --lr 1.5e-5 --generation \
#                 --use_softmax --loss_softmax_lr 1.5e-5 --fc_lr 2e-5 \
#                 --gen_lr 1e-4 --dis_lr 1e-4

# # best hdml & D1 & J_fan
# python main_s1_rev.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
#                 --n_epochs 120 --project THSG --group CARS_THSG_S1_D1_Jfan --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
#                 --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
#                 --split_subset_perc 0.25 --eval_frq 5 \
#                 --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
#                 --log_online --lr 1.5e-5 --generation \
#                 --use_softmax --loss_softmax_lr 1.5e-5 --fc_lr 2e-5 \
#                 --gen_lr 5e-4 --dis_lr 5e-4

# python main_s1_rev.py --dataset cars196 --kernels 6 --source "/root/Metric Learning/RetrievalHardGeneration/data/" --evaluate_on_gpu \
#                 --n_epochs 120 --project THSG --group CARS_THSG_S1_D1_Jfan --seed 0 --gpu 0 --bs 126 --samples_per_class 7 \
#                 --loss triplet --batch_mining semihard --arch resnet50_frozen_normalize \
#                 --split_subset_perc 0.25 --eval_frq 5 \
#                 --evaluation_metrics 'e_recall@1' 'e_recall@2' 'e_recall@4' 'nmi' 'f1' 'dists@intra' 'dists@inter' 'dists@intra_over_inter' \
#                 --log_online --lr 1.5e-5 --generation \
#                 --use_softmax --loss_softmax_lr 1.5e-5 --fc_lr 2e-5 \
#                 --gen_lr 1e-3 --dis_lr 1e-3