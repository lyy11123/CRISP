generate_exp:
	rm -rf ./cfg ./py
	mkdir -p ./cfg ./py ./results ./pic
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

draw_pic:
	mkdir -p ./pic
	python multifaults_pic.py

clean:
	rm -rf ./cfg ./py ./results ./pic
