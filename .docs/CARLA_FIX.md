# CARLA Import Fix

## Problem
CARLA 0.9.15 is compiled for Python 3.7, but the devcontainer was using Python 3.10. This caused:
1. Missing `libjpeg.so.8` library
2. Binary incompatibility between Python versions (segmentation fault)

## Solution Applied

### 1. Updated Dockerfile
The [.devcontainer/Dockerfile](.devcontainer/Dockerfile) has been updated to:
- Use Python 3.7 base image instead of Python 3.10
- Install `libjpeg8` and `libjpeg-turbo8` dependencies
- Install the correct CARLA wheel (`cp37-cp37m-manylinux_2_27_x86_64.whl`)

### 2. Next Steps
To apply the fix, you need to **rebuild the devcontainer**:

1. In VSCode, press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Linux/Windows)
2. Type "Dev Containers: Rebuild Container"
3. Select the option and wait for the rebuild to complete

## Verification
After rebuilding, verify CARLA works:
```bash
python -c "import carla; print('✅ CARLA client works!')"
```

## What Was Changed

### [.devcontainer/Dockerfile](.devcontainer/Dockerfile:2-3)
```dockerfile
# Changed from python:3.10-bullseye to python:3.7-bullseye
FROM --platform=linux/amd64 python:3.7-bullseye
```

### [.devcontainer/Dockerfile](.devcontainer/Dockerfile:36-41)
```dockerfile
# Added libjpeg8 installation
RUN wget http://security.ubuntu.com/ubuntu/pool/main/libj/libjpeg-turbo/libjpeg-turbo8_2.0.3-0ubuntu1_amd64.deb \
    && wget http://security.ubuntu.com/ubuntu/pool/main/libj/libjpeg8-empty/libjpeg8_8c-2ubuntu8_amd64.deb \
    && dpkg -i libjpeg-turbo8_2.0.3-0ubuntu1_amd64.deb \
    && dpkg -i libjpeg8_8c-2ubuntu8_amd64.deb \
    && rm *.deb
```

### [.devcontainer/Dockerfile](.devcontainer/Dockerfile:78)
```dockerfile
# Changed to use cp37 wheel instead of cp310
RUN pip install --no-cache-dir /opt/carla/PythonAPI/carla/dist/carla-${CARLA_VERSION}-cp37-cp37m-manylinux_2_27_x86_64.whl
```

## Temporary Workaround (Current Session Only)
The `libjpeg8` library has been installed in the current container, but Python version mismatch still prevents CARLA from working. You must rebuild the container to get Python 3.7.

To manually install libjpeg8 in future sessions (if needed):
```bash
wget http://security.ubuntu.com/ubuntu/pool/main/libj/libjpeg-turbo/libjpeg-turbo8_2.0.3-0ubuntu1_amd64.deb
wget http://security.ubuntu.com/ubuntu/pool/main/libj/libjpeg8-empty/libjpeg8_8c-2ubuntu8_amd64.deb
sudo dpkg -i libjpeg-turbo8_2.0.3-0ubuntu1_amd64.deb
sudo dpkg -i libjpeg8_8c-2ubuntu8_amd64.deb
```
