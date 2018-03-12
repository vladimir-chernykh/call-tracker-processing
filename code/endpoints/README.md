# Endpoints

Below there is an up-to-date description of the currently available API endpoints.

# /content

Endpoint `/content` is responsible for working with data at the server. Using this endopint one can upload or delete file.

### Upload

To upload a file one should submit a POST request with a file in the form field.

<code class="prettyprint"><strong>POST</strong> \<address\>/content</code>
  
Example of request:

```bash
curl <address>/content -X POST -F audio=@./data/examples/c-dur.mp3
```

### Delete

To delete a file one should submit a DELETE request to the child endpoint.

<code class="prettyprint"><strong>DELETE</strong> \<address\>/content/\<content_id\></code>

Here `<content_id>` is the 'content_id' which is returned by the uploading operation.
  
Example of request:

```bash
curl <address>/content/<content_id> -X DELETE
```

# /duration

This endpoint `/duration` computes the audio file duration.

<code class="prettyprint"><strong>POST</strong> \<address\>/duration</code>

One can specify the file for the analysis in two ways:

* Upload directly using form field ot the request
```bash
curl <address>/duration -X POST -F audio=@./data/examples/c-dur.mp3
```
* Use the 'content_id' which is returned by the `content` endpoint while uploading data to the server
```bash
curl <address>/content -X POST -F content_id=<content_id>
```
