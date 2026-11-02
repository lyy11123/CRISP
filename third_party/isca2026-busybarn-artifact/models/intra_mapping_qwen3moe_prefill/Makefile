generate_exp:
	rm -rf ./cfg ./py
	mkdir -p ./cfg ./py ./results
	python cfg.py
	python ln.py
	python mha.py
	python proj.py
	python ffn.py
	chmod +x ./run.sh

run_scripts:
	mkdir -p ./results
	bash ./run.sh

run_slurm:
	bash ./run_slurm.sh

clean_results:
	rm -rf ./results

clean:
	rm -rf ./cfg ./py ./results
