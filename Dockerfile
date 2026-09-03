# =============================================================================
#  rl_quad_traj_project — GPU development environment
# =============================================================================
#  This image is a "toolbox": it contains the compilers, CUDA, system libraries
#  and Python environments needed to BUILD and RUN every repository in this
#  project. The repositories themselves are NOT baked into the image — they live
#  in ./repos on your computer and are mounted into the container so you can
#  edit them in your normal editor and rebuild from source. See README.md.
#
#  Four isolated Python environments are created (their deps conflict):
#    /opt/venvs/main      (Python 3.12) -> crazyflow, gym-pybullet-drones,
#                                          lsy_drone_racing, RAPTOR_in_RotorPy,
#                                          crazy_track (racing task, training side)
#    /opt/venvs/race      (Python 3.12) -> lsy_drone_racing + ITS OWN crazyflow
#                                          (racing task, race side — see
#                                          scripts/setup_python_envs.sh for why)
#    /opt/venvs/crazysim  (Python 3.11) -> CrazySim cflib/cfclient (numpy<1.25)
#    /opt/venvs/datt      (Python 3.10) -> DATT (legacy 2022 stack)
#
#  C++ repos (learning-to-fly, raptor) use no Python env — they build with CMake.
# =============================================================================

# ---- GPU (default) vs CPU-only image ----------------------------------------
#  GPU (default): the NVIDIA CUDA base + jax[cuda12] + CUDA PyTorch.
#  CPU-only: build a smaller image with no CUDA. From the build_cpu.sh helper, or:
#      docker build \
#        --build-arg BASE_IMAGE=ubuntu:22.04 \
#        --build-arg MAIN_REQS=main-cpu.txt \
#        --build-arg DATT_REQS=datt-cpu.txt \
#        --build-arg MAIN_CONSTRAINTS=constraints-cpu.txt \
#        -t rl-quad-traj:cpu .
ARG BASE_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# Which requirements/constraints files to use (overridden for the CPU image).
ARG MAIN_REQS=main.txt
ARG DATT_REQS=datt.txt
ARG MAIN_CONSTRAINTS=constraints.txt

# ---- Optional heavy components (toggle at build time) -----------------------
#   --build-arg INSTALL_GAZEBO=true  to add Gazebo Garden for CrazySim
#   (we use MuJoCo by default, so Gazebo is OFF to keep the image smaller)
ARG INSTALL_GAZEBO=false

# ---- System packages --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        # build toolchain (C++ repos: learning-to-fly, raptor, crazyflie firmware)
        build-essential cmake ninja-build git wget curl ca-certificates \
        gnupg lsb-release pkg-config sudo nano less unzip \
        # C++ library deps used by RLtools / learning-to-fly
        libhdf5-dev libboost-all-dev protobuf-compiler libprotobuf-dev \
        libopenblas-dev \
        # OpenGL / GLFW / EGL — MuJoCo & PyBullet rendering
        libgl1-mesa-glx libgl1-mesa-dri libglfw3 libglew-dev libosmesa6-dev \
        libegl1 libgles2 mesa-utils patchelf \
        # X11 (so GUIs / viewers can show on the host display)
        libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcb-cursor0 \
        x11-apps libqt5gui5 \
        # `python` shim — the CrazySim firmware's version.c generator calls `python`
        python-is-python3 \
        # python (system 3.10) + 3.11/3.12 from deadsnakes.
        #   main env -> 3.12 (learnsyslab/crazyflow uses `value in EnumClass`, which
        #               only works on Python >= 3.12)
        #   crazysim -> 3.11 ; datt -> 3.10
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev \
        python3.11 python3.11-venv python3.11-dev \
        python3.12 python3.12-venv python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

# ---- Optional: Gazebo Garden (CrazySim full SITL) ---------------------------
RUN if [ "$INSTALL_GAZEBO" = "true" ]; then \
        curl https://packages.osrfoundation.org/gazebo.gpg \
            --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
            > /etc/apt/sources.list.d/gazebo-stable.list && \
        apt-get update && apt-get install -y --no-install-recommends gz-garden && \
        rm -rf /var/lib/apt/lists/* ; \
    fi

# Make the NVIDIA libraries visible (works only on machines with an NVIDIA GPU;
# on macOS these are simply ignored and everything runs on the CPU).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    # MuJoCo: use EGL for headless GPU rendering when a GPU is present
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    # drone-models / drone-controllers require the scipy array-API flag to be
    # set BEFORE scipy is imported, or they raise at import time.
    SCIPY_ARRAY_API=1

# -----------------------------------------------------------------------------
#  User + ownership — create the dev user and hand it /opt/venvs BEFORE building
#  the venvs, so every venv file is written as `dev`. This avoids a final
#  `chown -R /opt/venvs`, which would duplicate ~13 GB of venv data into an extra
#  image layer. UID/GID match the host (pass HOST_UID/HOST_GID at build time).
# -----------------------------------------------------------------------------
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} dev 2>/dev/null || true && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash dev 2>/dev/null || true && \
    usermod -aG sudo dev && \
    echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /opt/venvs && chown ${UID}:${GID} /opt/venvs && \
    echo 'alias activate-main="source /opt/venvs/main/bin/activate"' >> /etc/bash.bashrc && \
    echo 'alias activate-race="source /opt/venvs/race/bin/activate"' >> /etc/bash.bashrc && \
    echo 'alias activate-datt="source /opt/venvs/datt/bin/activate"' >> /etc/bash.bashrc && \
    echo 'alias activate-crazysim="source /opt/venvs/crazysim/bin/activate"' >> /etc/bash.bashrc && \
    echo 'source /opt/venvs/main/bin/activate' >> /etc/bash.bashrc

USER dev

# -----------------------------------------------------------------------------
#  Python environments (created AS the dev user — no recursive chown needed)
#  We pre-install the heavy wheels here so the big downloads happen once at
#  build time. The repos are editable-installed LATER (at runtime) against
#  these environments — see scripts/setup_python_envs.sh.
# -----------------------------------------------------------------------------
# Copy each requirements file just before its env so editing one env's deps
# doesn't bust the Docker cache for the others.

# ---- main env (Python 3.12 — required by learnsyslab/crazyflow) -------------
# The constraints file keeps jax/jaxlib and the CUDA plugin on one consistent
# version (installing repos later can otherwise upgrade jaxlib past the plugin,
# which breaks GPU linear algebra: "No FFI handler registered for cusolver...").
COPY requirements/${MAIN_REQS} /tmp/requirements/main.txt
COPY requirements/${MAIN_CONSTRAINTS} /tmp/requirements/constraints.txt
RUN python3.12 -m venv /opt/venvs/main && \
    /opt/venvs/main/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venvs/main/bin/pip install --no-cache-dir -c /tmp/requirements/constraints.txt \
        -r /tmp/requirements/main.txt

# ---- datt env (Python 3.10, legacy stack) -----------------------------------
# gym==0.21 only builds with old setuptools/wheel, so pin those first.
COPY requirements/${DATT_REQS} /tmp/requirements/datt.txt
RUN python3.10 -m venv /opt/venvs/datt && \
    /opt/venvs/datt/bin/pip install --no-cache-dir "pip<24.1" "setuptools==65.5.0" "wheel==0.38.4" && \
    /opt/venvs/datt/bin/pip install --no-cache-dir -r /tmp/requirements/datt.txt

# ---- race env (Python 3.12, racing task) -------------------------------------
# Starts bare: scripts/setup_python_envs.sh (or tasks/racing/setup.sh) installs
# lsy_drone_racing + PyPI crazyflow, held to the SAME versions as main by
# requirements/constraints*.txt, plus the CPU policy stack. Kept separate
# from `main` because crazy_track and lsy_drone_racing may pin DIFFERENT
# crazyflow versions — sharing one env can silently downgrade the simulator
# under a policy you already trained.
RUN python3.12 -m venv /opt/venvs/race && \
    /opt/venvs/race/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- crazysim env (Python 3.11) — CrazySim cflib/cfclient need numpy<1.25 ----
# Isolated because that numpy pin conflicts with the main env (crazyflow needs >=2).
COPY requirements/crazysim.txt /tmp/requirements/crazysim.txt
RUN python3.11 -m venv /opt/venvs/crazysim && \
    /opt/venvs/crazysim/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venvs/crazysim/bin/pip install --no-cache-dir -r /tmp/requirements/crazysim.txt

WORKDIR /workspace
CMD ["/bin/bash"]
