FROM mitdrl/ubuntu:latest

WORKDIR /env

COPY ./environment.yml /env

# Update the conda base environment
RUN conda env update --name base --file /env/environment.yml
