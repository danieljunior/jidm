# JIDM submission

## Execução 

#### Construção da imagem
- `docker build -t jidm .`

#### Iniciar containers

- `docker run --rm --shm-size="2g" -v ${PWD}:/app -w /app --name jidm -itd jidm bash`

#### Acessar container
- `docker exec -it jidm bash`