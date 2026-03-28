preprocess_dataset:
	python preprocess_speeches.py
	python generate_aligenment_dataset.py

train_model:
	sbatch deploy_job.sh


deploy_model:
	sbatch deploy_model.sh