FROM cpllab/language-models:ordered-neurons

ARG MODEL_ROOT=models/acl-ordered-neurons

RUN rm -r ${LMZOO_CHECKPOINT_PATH}/*

COPY ${MODEL_ROOT}/get_surprisals.py /opt/Ordered-Neurons
COPY ${MODEL_ROOT}/get_raw.py /opt/Ordered-Neurons
COPY ${MODEL_ROOT}/bin/tokenize /opt/bin

ENV LMZOO_CHECKPOINT_PATH /opt/Ordered-Neurons/checkpoint
ENV LMZOO_VOCABULARY_PATH vocab

# Current git commit of build repository
ARG COMMIT
# sha1 checksum of build directory
ARG FILES_SHA1

# Prepare spec.
COPY ${MODEL_ROOT}/spec.template.json /tmp/spec.template.json
RUN export BUILD_DATETIME="$(date)"
RUN cat /tmp/spec.template.json | \
            sed "s/<image\.sha1>/$COMMIT/g; s/<image\.files_sha1>/${FILES_SHA1}/g; s/<image\.datetime>/${BUILD_DATETIME}/g" \
            > /opt/spec.template.json

WORKDIR /opt/bin
