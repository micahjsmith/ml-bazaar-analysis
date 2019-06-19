.DEFAULT_GOAL := main

IMAGE_NAME := mlbazaar2019
EXPLORE_NAME := mlbazaar2019_explore

help:
	@echo "Usage:"
	@echo "  help       	Show this message"
	@echo "  main       	Run experiments"
	@echo "  clean      	Remove docker containers and script output files"
	@echo "  clean-all	make clean, and also remove docker images"

install:
	@command -v docker || { echo "Could not find docker executable, please install it: https://docs.docker.com/install/"; exit 1; }

build:
	docker build -t $(IMAGE_NAME) .
	docker build -t $(EXPLORE_NAME) -f Dockerfile-explore .
	mkdir -p ./output

main: install build
	docker run \
	    --rm \
	    --tty \
	    --mount "type=bind,src=$(shell pwd)/output/,dst=/work/output/" \
	    --mount "type=bind,src=$(shell pwd)/data/,dst=/work/data/" \
	    $(IMAGE_NAME)

explore: install build
	docker run \
	    -i \
	    --tty \
	    --mount "type=bind,src=$(shell pwd)/output/,dst=/work/output/" \
	    --mount "type=bind,src=$(shell pwd)/data/,dst=/work/data/" \
	    -p 8888:8888 \
	    $(EXPLORE_NAME)

clean:
	docker rm -f $(shell docker ps -a -q --filter ancestor=$(IMAGE_NAME)) 2>/dev/null || true
	rm -rf ./output
	rm -rf ./data/cache
	rm -rf ./__pycache__
	rm -rf ./.ipynb_checkpoints


clean-all: clean
	docker rmi -f $(IMAGE_NAME) 2>/dev/null || echo "No images found"
