FROM tensorflow/tensorflow

# Dependencies
RUN apt-get update && apt-get install git -y
RUN pip3 install -U pylint
