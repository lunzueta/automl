FROM tensorflow/tensorflow

# Dependencies
RUN apt-get update && apt-get install git -y
RUN pip3 install -U pylint && pip3 install tensorflow-datasets && \
    pip3 install pyyaml && pip3 install tensorflow-addons
