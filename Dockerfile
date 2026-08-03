FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /pathclip


COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .