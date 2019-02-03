.DEFAULT_GOAL := main

IMAGE_NAME := mlbazaar2019

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
	mkdir -p ./output

main: install build
	docker run \
	    --rm \
	    --tty \
	    --mount "type=bind,src=$(shell pwd)/output/,dst=/tmp/output/" \
	    --mount "type=bind,src=$(shell pwd)/data/,dst=/tmp/data/" \
	    $(IMAGE_NAME)

explore: install build
	docker run \
	    -i \
	    --tty \
	    --mount "type=bind,src=$(shell pwd)/output/,dst=/tmp/output/" \
	    --mount "type=bind,src=$(shell pwd)/data/,dst=/tmp/data/" \
	    $(IMAGE_NAME) \
	    ipython

clean:
	docker rm -f $(shell docker ps -a -q --filter ancestor=$(IMAGE_NAME)) 2>/dev/null || true
	rm -rf ./output

clean-all: clean
	docker rmi -f $(IMAGE_NAME) 2>/dev/null || echo "No images found"
