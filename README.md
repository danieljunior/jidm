# JIDM submission

## Execução 

#### Construção da imagem
- `docker build -t jidm .`

#### Iniciar containers

- `docker run --rm --shm-size="2g" -v ${PWD}:/app -w /app -p 8888:8888 --name jidm -itd jidm bash`

#### Acessar container
- `docker exec -it jidm bash`

#### Iniciar jupyter notebook
- `jupyter lab --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token='jidm' &`

#### Acessar o jupyter notebook
- http://localhost:8888/lab
