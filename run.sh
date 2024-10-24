# conformal baseline
# clean conformal
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_baseline -calibrate -alpha 0.1 -dataset GTSRB -sigma 0.12 -attack_type none -seed 2023 -score_type hps
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_baseline -inference -alpha 0.1 -dataset GTSRB -sigma 0.12 -attack_type none -seed 2023 -score_type hps -max_norm 0.125
# pgd attack conformal
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_baseline -calibrate -alpha 0.1 -dataset GTSRB -sigma 0.5 -attack_type pgd -seed 2023
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_baseline -inference -alpha 0.1 -dataset GTSRB -sigma 0.5 -attack_type pgd -seed 2023


# conformal knowledge probabilistic circuit (PC)
# clean conformal
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc -knowledge_weights 1.0 -pc_weight 1.0 -calibrate -alpha 0.1 -dataset GTSRB -sigma 0.12 -attack_type none -seed 2023
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc -knowledge_weights 1.0 -pc_weight 1.0 -inference -alpha 0.1 -dataset GTSRB -sigma 0.12 -attack_type none -seed 2023
# pgd attack conformal
#CUDA_VISIBLE_DEVICES=6 python main.py -load_exist 1 -method conformal_knowledge_pc -sigma 0.12 -knowledge_weights 1.0 -pc_weight 1.0 -calibrate -alpha 0.1 -dataset GTSRB -attack_type pgd -seed 2023 -score_type hps
#CUDA_VISIBLE_DEVICES=6 python main.py -load_exist 1 -method conformal_knowledge_pc -knowledge_weights 1.0 -sigma 0.12 -knowledge_set_correction 0 -pc_weight 1.0 -inference -alpha 0.1 -dataset GTSRB -attack_type pgd -seed 2023 -max_norm 0.125 -score_type hps

#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc_ori_score -knowledge_weights 10.0 -pc_weight 1.0 -calibrate -alpha 0.1 -dataset GTSRB -sigma 0.12 -attack_type none -seed 2023 -max_norm 0.5
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc_ori_score -knowledge_weights 10.0 -pc_weight 1.0 -inference -alpha 0.1 -dataset GTSRB -sigma 0.12 -attack_type pgd -seed 2023 -max_norm 0.5


# conformal knowledge probabilistic circuit (PC) certified robustness
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc -pc_correction 1 -knowledge_weights 1.0 -pc_weight 1.0 -calibrate -alpha 0.1 -dataset GTSRB -sigma 0.12 -attack_type certify -seed 2023
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc -pc_correction 1 -knowledge_weights 1.0 -pc_weight 1.0 -num_certify 50 -knowledge_set_correction 1 -sigma 0.12 -sigma_certify 0.5 -inference -alpha 0.1 -dataset GTSRB  -attack_type certify -seed 2023 -batch_size 1


# conformal baseline certified robustness
#CUDA_VISIBLE_DEVICES=3 python main.py -method conformal_baseline -num_certify 100 -calibrate -alpha 0.1 -dataset cifar10 -sigma 0.5 -sigma_certify 0.5 -attack_type certify -seed 2023 -batch_size 1 -score_type aps
#CUDA_VISIBLE_DEVICES=3 python main.py -method conformal_baseline -num_certify 100 -inference -alpha 0.1 -dataset GTSRB -sigma 0.5 -sigma_certify 0.5 -attack_type certify -max_norm 0.25 -seed 2023 -score_type aps

# worst-case coverage calculation of ours
#CUDA_VISIBLE_DEVICES=2 python main.py -method conformal_knowledge_pc_worst_cov -num_certify 100 -N_certify 10000 -pc_correction 1 -knowledge_weights 1.0 -pc_weight 1.0 -calibrate -alpha 0.1 -dataset cifar10 -sigma 0.5 -sigma_certify 1.0 -seed 2023 -batch_size 1

# worst-case converage of baseline on AwA
#CUDA_VISIBLE_DEVICES=5 python AwA2_single_model_certify.py

# conformal PC knowledge on AwA
#CUDA_VISIBLE_DEVICES=5 python AwA_conformal_knowledge_PC.py

# train PC
#CUDA_VISIBLE_DEVICES=4 python main.py -method train_PC

#CUDA_VISIBLE_DEVICES=8 python main.py -load_exist 1 -knowledge_set_correction 0 -method conformal_knowledge_pc -calibrate -alpha 0.1 -dataset GTSRB -sigma 0.12 -attack_type pgd -seed 2023
#CUDA_VISIBLE_DEVICES=8 python main.py -load_exist 1 -knowledge_set_correction 0 -method conformal_knowledge_pc -inference -alpha 0.1 -dataset GTSRB -sigma 0.12 -attack_type pgd -max_norm 0.0 -seed 2023


# Evaluation on CIFAR-10
# conformal baseline
# clean conformal
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_baseline -calibrate -alpha 0.1 -dataset cifar10 -sigma 0.0 -attack_type none -seed 2023
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_baseline -inference -alpha 0.1 -dataset cifar10 -sigma 0.0 -attack_type none -seed 2023
# pgd attack conformal
#CUDA_VISIBLE_DEVICES=6 python main.py -method conformal_baseline -calibrate -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type pgd -seed 2023 -max_norm 0.125
#CUDA_VISIBLE_DEVICES=6 python main.py -method conformal_baseline -inference -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type pgd -seed 2023 -max_norm 0.125

# conformal knowledge probabilistic circuit (PC)
# clean conformal
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc -knowledge_weights 1.0 -pc_weight 1.0 -calibrate -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type none -seed 2023
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc -knowledge_weights 1.0 -pc_weight 1.0 -inference -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type none -seed 2023
# pgd attack conformal
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc -knowledge_weights 1.0 -pc_weight 1.0 -calibrate -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type pgd -seed 2023 -max_norm 0.25
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc -knowledge_weights 1.0 -pc_weight 1.0 -inference -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type pgd -seed 2023 -max_norm 0.25


# conformal knowledge probabilistic circuit (PC)
# clean conformal
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc_ori_score -knowledge_weights 1.0 -pc_weight 1.0 -calibrate -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type none -seed 2023
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc_ori_score -knowledge_weights 1.0 -pc_weight 1.0 -inference -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type none -seed 2023
# pgd attack conformal
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc_ori_score -knowledge_weights 1.0 -pc_weight 1.0 -calibrate -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type pgd -seed 2023 -max_norm 0.5
#CUDA_VISIBLE_DEVICES=8 python main.py -method conformal_knowledge_pc_ori_score -knowledge_weights 1.0 -pc_weight 1.0 -inference -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type pgd -seed 2023 -max_norm 0.5



# conformal knowledge probabilistic circuit (PC) certified robustness
#CUDA_VISIBLE_DEVICES=4 python main.py -method conformal_knowledge_pc -pc_correction 1 -knowledge_weights 1.0 -pc_weight 1.0 -calibrate -alpha 0.1 -dataset cifar10 -sigma 0.12 -attack_type certify -seed 2023
#CUDA_VISIBLE_DEVICES=4 python main.py -method conformal_knowledge_pc -pc_correction 1 -knowledge_weights 1.0 -pc_weight 1.0 -num_certify 50 -N_certify 100000 -knowledge_set_correction 1 -sigma 0.12 -sigma_certify 0.6 -inference -alpha 0.1 -dataset cifar10 -attack_type certify -seed 2023 -batch_size 1


# conformal baseline certified robustness
#CUDA_VISIBLE_DEVICES=6 python main.py -method conformal_baseline -num_certify 50 -calibrate -alpha 0.1 -dataset cifar10 -sigma 0.12 -sigma_certify 1.0 -attack_type certify -seed 2023 -batch_size 1 -max_norm 0.5
#CUDA_VISIBLE_DEVICES=6 python main.py -method conformal_baseline -num_certify 50 -inference -alpha 0.1 -dataset cifar10 -sigma 0.12 -sigma_certify 1.0 -attack_type certify -seed 2023 -max_norm 0.5