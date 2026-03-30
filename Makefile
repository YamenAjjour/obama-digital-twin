setup_environment:
	module load Miniforge3
	conda activate
	pip install -r requirements.txt

preprocess_dataset:
	python preprocess_speeches.py
	python generate_aligenment_dataset.py

train_model:
	sbatch deploy_job.sh

push_model:
	hf auth login
	hf upload yamenajjour/obama-digital-twin qwen_dpo_lora_output

deploy_model:
	sbatch deploy_model.sh