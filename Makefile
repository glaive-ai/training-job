export IMAGE_NAME=glaiveai/training-container:$(shell git rev-parse HEAD)
export JOB_NAME=training-job-$(shell git rev-parse HEAD)


.PHONY: build
build:
	sudo docker build -f Dockerfile --build-arg IMAGE_NAME=$(IMAGE_NAME) ./ -t $(IMAGE_NAME)

.PHONY: push
push: build
	sudo docker push $(IMAGE_NAME)

.PHONY: launch-job 
launch-job: 
	envsubst < $(config) | kubectl apply -f -

.PHONY: delete-job
delete-job:
	envsubst < $(config)  | kubectl delete -f -

.PHONY: launch-docker
launch-docker:
	sudo docker run --gpus '"device=0,1,2,3,4,5,6,7"'  --rm -it --entrypoint bash -e GOOGLE_APPLICATION_CREDENTIALS=gcs.json -e WANDB_API_KEY=a07f9a332409243f4cd7eecffd733b9297f9436e -v $(shell pwd):/workspace $(IMAGE_NAME) 
