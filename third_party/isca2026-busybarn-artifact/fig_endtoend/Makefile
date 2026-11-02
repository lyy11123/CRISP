run_models:
	@echo "Running all 12 model experiments..."
	@for d in ../models/*/; do \
		echo "=== $$d ==="; \
		(cd "$$d" && make generate_exp && make run_scripts); \
	done
	@echo "All model experiments done."

run_slurm:
	bash ./run_slurm.sh

run_aggregate:
	python end_to_end.py

draw_pic:
	python endtoend_pic.py

all: run_models run_aggregate draw_pic

clean_models:
	@for d in ../models/*/; do \
		(cd "$$d" && make clean_results 2>/dev/null); \
	done

clean:
	rm -rf ./results ./pic
	@for d in ../models/*/; do \
		(cd "$$d" && make clean 2>/dev/null); \
	done
