FROM cpllab/language-models:ngram

ARG MODEL_ROOT=models/acl-ngram

# Remove checkpoint -- this will be overwritten anyway.
RUN rm -r ${LMZOO_CHECKPOINT_PATH}/*

# Runtime dependencies
RUN pip install nltk

COPY ${MODEL_ROOT}/get_raw.py /opt/srilm
COPY ${MODEL_ROOT}/bin/tokenize /opt/bin

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
