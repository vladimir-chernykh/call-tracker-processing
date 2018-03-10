# call-tracker-processing
This is the part of Call Tracker project which process all the calls and outputs the statistics

# Commands to launch:
* Build a docker image with:
```
docker build -f Dockerfile -t call-tracker-processing .
```
* Launch docker container:
```
docker run --rm -d -p 3000:3000 call-tracker-processing
```
* Do a post request:
```
curl "localhost:3000/get_duration" -X POST -F "audio=@./data/examples/c-dur.mp3"
```
