export IMAGE_NAME=glaiveai/training-container:$(shell git rev-parse HEAD)
export JOB_NAME=training-job-$(shell git rev-parse HEAD)


.PHONY: build
build:
	sudo docker build -f Dockerfile ./ -t $(IMAGE_NAME)

.PHONY: push
push: build
	sudo docker push $(IMAGE_NAME)

.PHONY: launch-job
launch-job: 
	envsubst < configs/mistral7B-oss-instruct.yaml | kubectl apply -f -

.PHONY: delete-job
delete-job:
	envsubst < configs/mistral7B-oss-instruct.yaml | kubectl delete -f -

.PHONY: launch-local
launch-local:
	sudo docker run --gpus all --rm -it --entrypoint bash -e GOOGLE_APPLICATION_CREDENTIALS=gcs.json -e WANDB_API_KEY=a07f9a332409243f4cd7eecffd733b9297f9436e -v $(shell pwd):/workspace $(IMAGE_NAME) 