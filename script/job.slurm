#!/bin/bash
#SBATCH --job-name=afner         # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --ntasks=1              # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --constraint=a100        # for running on an A100 GPU
#SBATCH --constraint=gpu80       # for an 100 with 80 GB of memory
#SBATCH --time=8:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=hb3815@princeton.edu

# which gpu node was used
echo "Running on host" $(hostname)

# print the slurm environment variables sorted by name
printenv | grep -i slurm | sort

module purge
module load anaconda3/2023.9
conda activate /home/hb3815/.conda/envs/torch-env


for LANG in efi
do
	for j in 1 2 3 4 5
	do
		export MAX_LENGTH=200
		export BERT_MODEL=castorini/afriberta_large
		export OUTPUT_DIR=baseline_models/${LANG}_afriberta
		export TEXT_RESULT=test_result$j.txt
		export TEXT_PREDICTION=test_predictions$j.txt
		export BATCH_SIZE=16
		export NUM_EPOCHS=2
		export SAVE_STEPS=10
		export SEED=$j

		#CUDA_VISIBLE_DEVICES=3
		HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
		python3 ../train_pos.py --data_dir ../data/${LANG}/ \
		--model_type xlmroberta \
		--model_name_or_path $BERT_MODEL \
		--output_dir $OUTPUT_DIR \
		--test_result_file $TEXT_RESULT \
		--test_prediction_file $TEXT_PREDICTION \
		--max_seq_length  $MAX_LENGTH \
		--num_train_epochs $NUM_EPOCHS \
		--per_gpu_train_batch_size $BATCH_SIZE \
		--save_steps $SAVE_STEPS \
		--gradient_accumulation_steps 2 \
		--seed $SEED \
		--do_train \
		--do_eval \
		--do_predict \
		--overwrite_output_dir
	done
done
