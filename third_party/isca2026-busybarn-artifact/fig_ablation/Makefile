generate_cfg:
	python cfg.py

run_experiments:
	mkdir -p ./results
	chmod +x ./run.sh
	bash ./run.sh

run_slurm:
	bash ./run_slurm.sh

draw_pic:
	python plot_16x16_1x1.py
	python plot_pie_time.py

clean:
	rm -rf ./results ./sbatch_log ./cfg
