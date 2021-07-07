FROM nvcr.io/nvidia/pytorch:21.06-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

ENV USER_NAME aicrowd
ENV HOME_DIR /home/$USER_NAME

ENV HOST_UID 1001
ENV HOST_GID 1001

RUN adduser --disabled-password \
            --gecos "Default user" \
            --uid ${HOST_UID} \
            ${USER_NAME}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

RUN pip install opencv-python toolz
RUN pip install nle aicrowd-gym

COPY --chown=1001:1001 . ${HOME_DIR}
