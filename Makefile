update-env:
	conda env export | grep -v "prefix" > envs/conda_env.yml
.PHONY: update_env
