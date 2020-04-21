.DEFAULT_GOAL := main

IMAGE_NAME := mlbazaar2019
HOST_PORT := 8888
CONTAINER_PORT := 8888

help:
	@echo "Usage:"
	@echo "  help       	Show this message"
	@echo "  main       	Run experiments"
	@echo "  clean      	Remove docker containers and script output files"
	@echo "  clean-all	make clean, and also remove docker images"

install:
	@command -v docker || { echo "Could not find docker executable, please install it: https://docs.docker.com/install/"; exit 1; }

build: install
	docker build -t $(IMAGE_NAME) .
	mkdir -p ./output

main: build
	docker run \
	    --rm \
	    --tty \
	    --mount "type=bind,src=$(shell pwd)/output/,dst=/work/output/" \
	    --mount "type=bind,src=$(shell pwd)/data/,dst=/work/data/" \
	    $(IMAGE_NAME)

explore: build
	docker run \
	    --rm \
	    -i \
	    --tty \
	    --mount "type=bind,src=$(shell pwd)/output/,dst=/work/output/" \
	    --mount "type=bind,src=$(shell pwd)/data/,dst=/work/data/" \
	    --mount "type=bind,src=$(shell pwd)/notebooks/,dst=/work/notebooks/" \
	    -p $(HOST_PORT):$(CONTAINER_PORT) \
	    $(IMAGE_NAME) \
	    jupyter lab --ip=0.0.0.0 --port=$(CONTAINER_PORT) --allow-root --no-browser --LabApp.token=

clean:
	-docker rm -f $(shell docker ps -a -q --filter ancestor=$(IMAGE_NAME)) 2>/dev/null
	rm -rf ./output
	rm -rf ./data/cache
	rm -rf ./__pycache__
	rm -rf ./.ipynb_checkpoints


clean-all: clean
	docker rmi -f $(IMAGE_NAME) 2>/dev/null || echo "No images found"
