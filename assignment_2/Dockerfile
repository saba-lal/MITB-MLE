# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1-python3.10

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Debian 11 (the base image's OS) is end-of-life: its apt security repo now 404s, so
# 'apt-get install openjdk-17' fails. Instead, copy a ready-made Java 17 from the
# official Temurin image (multi-arch, so it also works on Apple Silicon).
COPY --from=eclipse-temurin:17-jdk /opt/java/openjdk /opt/java/openjdk

# procps provides 'ps' (comes from the main repo, which still works). The base image
# ships a PostgreSQL apt source that now 404s and breaks 'apt-get update'; drop it first.
RUN (sed -i '/apt.postgresql.org/d' /etc/apt/sources.list /etc/apt/sources.list.d/* 2>/dev/null || true) && \
    apt-get update && \
    apt-get install -y --no-install-recommends procps && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark's scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh

# Point Spark at Java
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a volume mount point for notebooks
VOLUME /app
