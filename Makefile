.PHONY: create-venv install-deps delete-venv download-data-set delete-data-set

create-venv:
	@python3 -m venv .venv
	@echo "Virtual environment created in .venv"

install-deps: create-venv
	@.venv/bin/pip install -r requirements.txt
	@echo "Dependencies installed in the virtual environment"

delete-venv:
	@rm -rf .venv
	@echo "Virtual environment deleted"

download-data-set:
	curl -L -o dataset/disfa.zip https://www.kaggle.com/api/v1/datasets/download/abhishekg27/disfa
	@unzip dataset/disfa.zip -d dataset/
	@echo "Dataset downloaded and extracted to dataset/disfa"

delete-data-set:
	@rm -rf dataset/disfa.zip dataset/disfa
	@echo "Dataset deleted"