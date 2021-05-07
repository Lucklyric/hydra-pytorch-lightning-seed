.PHONY: update_env
update_env:
	conda env export | grep -v "prefix" > envs/conda_env.yml
