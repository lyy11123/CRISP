generate_exp:
	rm -rf ./cfg ./py
	mkdir -p ./cfg ./py ./results ./pic
	python cfg.py
	python cfg_fail.py
	python allgather_synthetic.py
	python alltoall_synthetic.py
	python allgather_fail.py
	python alltoall_fail.py
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
	python allgather_synthetic_pic.py
	python alltoall_synthetic_pic.py
	python fail_pic.py

clean:
	rm -rf ./cfg ./py ./results ./pic ./logs
