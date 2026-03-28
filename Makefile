preprocess_dataset:
	python preprocess_speeches.py
	python generate_aligenment_dataset.py

train_model:
	sbatch deploy_job.sh

push_model:
	hf auth login
    hf upload yamenajjour/obama-digita-twin qwen_dpo_lora_outpu

deploy_model:
	sbatch deploy_model.sh