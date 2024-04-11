# Datasets for Portuguese Legal Semantic Textual Similarity

## Running 

#### Build docker image
- `docker build -t jidm .`

#### Start containers

- `docker run --rm --shm-size="2g" -v ${PWD}:/app -w /app -p 8888:8888 --name jidm -itd jidm bash`

#### Access container
- `docker exec -it jidm bash`

#### Start jupyter notebook
- `jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='jidm' &`

#### Access jupyter notebook
- http://localhost:8888/lab
