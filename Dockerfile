# Base Image on DRL image
FROM mitdrl/ubuntu:latest

# Timezone
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get install -y libsndfile1
RUN conda install python=3.8 tensorflow-gpu tqdm
RUN pip install keras-ncp tensorflow-gpu==2.5.0 tensorflow-probability==0.13.0

# Update something to the bashrc (/etc/bashrc_skipper) to customize your shell
RUN pip install pyfiglet
RUN echo -e "alias py='python'" >> /etc/bashrc_skipper


# Switch to src directory
WORKDIR /src

# Copy your code into the docker that is assumed to live in . (on machine)
COPY ./ /src
