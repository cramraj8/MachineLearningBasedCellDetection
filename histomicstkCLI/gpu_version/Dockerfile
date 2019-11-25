
FROM cramraj8/base_docker_cuda:1.0
MAINTAINER Lee Cooper <lee.cooper@emory.edu>


# Copy plugin files and install any requirements
ENV my_cwd_path=$htk_path/../my_cwd
RUN mkdir -p $my_cwd_path && \
	apt-get update


COPY . $my_cwd_path


ENV luminoth_path=$htk_path/../my_cwd/luminoth
RUN cd $luminoth_path && \
    pip install -e . --ignore-installed PyYAML && \
    python setup.py install && \
    pip install opencv-contrib-python-headless


RUN cd $my_cwd_path && \
    pip install tensorflow-gpu==1.5.0


# use entrypoint provided by slicer_cli_web
WORKDIR $my_cwd_path/Applications
ENTRYPOINT ["/build/miniconda/bin/python" ,"/build/slicer_cli_web/server/cli_list_entrypoint.py"]
