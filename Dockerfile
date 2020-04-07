FROM debian:latest
MAINTAINER Conda Development Team <conda@continuum.io>

ADD environment.yml /tmp/environment.yml
# RUN conda env create -f /tmp/environment.yml
RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda env update --file /tmp/environment.yml \
    && conda install -y -c conda-forge igraph \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes   \
    && rm -rf /usr/local/share/terminfo/N/NCR260VT300WPP \
    && rm -rf /usr/local/pkgs/ncurses \
    && rm -rf /usr/local/share/terminfo/
# Workaround for singularity bug on mac  https://github.com/sylabs/singularity/issues/4301


ENV PATH /opt/conda/bin:$PATH

# FROM ubuntu:18.04
# MAINTAINER Soichi Hayashi <hayashis@iu.edu>



# ENV CONDADIR=/miniconda3/
# ENV CONDAPATH=$CONDADIR/bin/conda

# RUN apt-get update && apt-get install -y wget 
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# RUN chmod +x ./Miniconda3-latest-Linux-x86_64.sh
# RUN ./Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3



# RUN $CONDAPATH init --all

# ADD environment.yml /tmp/environment.yml
# RUN $CONDAPATH env create -f /tmp/environment.yml

# # Pull the environment name out of the environment.yml
# RUN echo "$CONDAPATH activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
# ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH
# RUN $CONDAPATH clean -afy
# RUN find $CONDADIR -follow -type f -name '*.a' -delete
# RUN find $CONDADIR -follow -type f -name '*.pyc' -delete
# RUN find $CONDADIR -follow -type f -name '*.js.map' -delete
# RUN find $CONDADIR -follow -type f -name '*.a' -delete


#RUN git clone https://github.com/nipy/dipy.git && cd dipy && pip install .
#ENV DEBIAN_FRONTEND=noninteractive

#make it work under singularity
RUN ldconfig && mkdir -p /N/u /N/home /N/dc2 /N/soft


#https://wiki.ubuntu.com/DashAsBinSh
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

ENV PYTHONNOUSERSITE=true





# FROM continuumio/miniconda3
# MAINTAINER Filipi Silva <hayashis@iu.edu>

# RUN conda create -n env python=3.7 numpy tqdm 
# RUN echo "source activate env" > ~/.bashrc
# ENV PATH /opt/conda/envs/env/bin:$PATH
# RUN conda clean -a
# RUN rm -r /tmp/*


# Pull the environment name out of the environment.yml
# RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
# ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

RUN find /usr/ -name NCR260VT300WPP -exec rm -f {} \;

RUN rm -rf /usr/local/share/terminfo/N/NCR260VT300WPP \
    && rm -rf /usr/local/pkgs/ncurses \
    && rm -rf /usr/local/share/terminfo/ 
