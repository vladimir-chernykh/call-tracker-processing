# Call Tracker Processing

This part of the ***Call Tracker*** project is responsible for audio files processing API. Processing here means extractions of various features and meta-information.

Up-to-date list of the endpoints can be found [here](code/endpoints/README.md)

# Fast API Launch

The commands below fire up the Docker container with API server running

* Clone a repo:
```sh
git clone git@github.com:vladimir-chernykh/call-tracker-processing.git
cd call-tracker-processing
```
* Build a docker image:
```sh
docker build -f Dockerfile -t call-tracker-processing .
```
* Launch docker container:
```sh
docker run --rm -d -p 3000:3000 call-tracker-processing
```
* Do a test post request:
```sh
curl localhost:3000/duration -X POST -F audio=@./data/examples/c-dur.mp3
```
