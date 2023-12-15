export IMAGE_NAME=glaiveai/training-container:$(shell git rev-parse HEAD)

.PHONY: build
build:
	sudo docker build -f Dockerfile ./ -t $(IMAGE_NAME)

.PHONY: push
push: build
	sudo docker push $(IMAGE_NAME)

.PHONY: launch-job
launch: 
	envsubst < training-job.yaml | kubectl apply -f -
