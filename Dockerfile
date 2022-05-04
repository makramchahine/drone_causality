FROM dolphonie1/causal_skipper:0.01

# add custom shap lib. This dockerfile should be built from the root of this repo
# with the shap dir inside
WORKDIR /src
RUN mkdir /src/drone_causality && mkdir /src/shap
COPY ./shap /src/shap
RUN pip install /src/shap

# install opencv deps for shap
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install seaborn

# add current repo contents
COPY . /src/drone_causality/
