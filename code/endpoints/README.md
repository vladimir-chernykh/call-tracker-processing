# Endpoints

Below there is an up-to-date description of the currently available API endpoints.

# /content

Endpoint `/content` is responsible for working with data at the server. Using this endopint one can upload or delete file.

### Upload

To upload a file one should submit a POST request with a file in the form field.

<code class="prettyprint"><strong>POST</strong> \<address\>/content</code>
  
**Response description**

JSON with 3 main fiedls:
* `status` - whether the call was executed successfully or not ("ok" or "error")
* `msg` - human-readable message about the status
* `results` - dictionary with the results of the call; it will be empty if the error has occured
  * `content_id` - unique identifier assigned to the loaded file
  
**Request example**

Input:

```bash
curl localhost:3000/content -X POST -F audio=@data/examples/c-dur.mp3
```

Output:

```json
{
  "msg": "The file has been added", 
  "result": {
    "content_id": "833b88aded804502aff4e665f805ec56"
  }, 
  "status": "ok"
}
```

### Delete

To delete a file one should submit a DELETE request to the child endpoint.

<code class="prettyprint"><strong>DELETE</strong> \<address\>/content/\<content_id\></code>

Here `<content_id>` is the 'content_id' which is returned by the uploading operation.

**Response description**

JSON with 3 main fiedls:
* `status` - whether the call was executed successfully or not ("ok" or "error")
* `msg` - human-readable message about the status
* `results` - dictionary with the results of the call; it will be empty if the error has occured
  * `content_id` - unique identifier of the deleted file
  
**Request example**

Input:

```bash
curl localhost:3000/content/<content_id> -X DELETE
```

Output:

```json
{
  "msg": "The file has been deleted", 
  "result": {
    "content_id": "833b88aded804502aff4e665f805ec56"
  }, 
  "status": "ok"
}
```

# /duration

This endpoint `/duration` computes the audio file duration.

<code class="prettyprint"><strong>POST</strong> \<address\>/duration</code>

**Response description**

JSON with 3 main fiedls:
* `status` - whether the call was executed successfully or not ("ok" or "error")
* `msg` - human-readable message about the status
* `results` - dictionary with the results of the call; it will be empty if the error has occured
  * `duration` - the duration of the audio in seconds

**Request example**

Input:

* Upload directly using form field ot the request
```bash
curl localhost:3000/duration -X POST -F audio=@data/examples/c-dur.mp3
```
* Use the 'content_id' which is returned by the `content` endpoint while uploading data to the server
```bash
curl localhost:3000/content -X POST -F content_id=<content_id>
```

Output:

```json
{
  "msg": "Duration has been calculated", 
  "result": {
    "duration": 4.56
  }, 
  "status": "ok"
}
```
