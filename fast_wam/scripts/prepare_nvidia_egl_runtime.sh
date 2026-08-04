#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/mnt/yuhan/FastWAM_megatron_robocasa_webdataset}"
DRIVER_VERSION="${NVIDIA_DRIVER_VERSION:-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')}"
DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
PACKAGE="libnvidia-gl-${DRIVER_MAJOR}"
PACKAGE_VERSION="${NVIDIA_PACKAGE_VERSION:-${DRIVER_VERSION}-0ubuntu1}"
RUNTIME="${NVIDIA_EGL_RUNTIME:-${ROOT}/.runtime/nvidia-egl-${DRIVER_VERSION}}"
DOWNLOAD="${RUNTIME}/download"
EXTRACTED="${RUNTIME}/root"
LIB="${EXTRACTED}/usr/lib/x86_64-linux-gnu"
DEB="${DOWNLOAD}/${PACKAGE}_${PACKAGE_VERSION}_amd64.deb"

mkdir -p "${DOWNLOAD}" "${EXTRACTED}"
if [[ ! -s "${DEB}" ]]; then
  (
    cd "${DOWNLOAD}"
    apt-get download "${PACKAGE}=${PACKAGE_VERSION}"
  )
fi
dpkg-deb -x "${DEB}" "${EXTRACTED}"

for library in \
  "${LIB}/libEGL_nvidia.so.${DRIVER_VERSION}" \
  "${LIB}/libnvidia-eglcore.so.${DRIVER_VERSION}"; do
  [[ -s "${library}" ]] || {
    echo "missing extracted NVIDIA EGL library: ${library}" >&2
    exit 2
  }
done

sha256sum \
  "${LIB}/libEGL_nvidia.so.${DRIVER_VERSION}" \
  "${LIB}/libnvidia-eglcore.so.${DRIVER_VERSION}" \
  > "${RUNTIME}/SHA256SUMS"
date -Is > "${RUNTIME}/DONE"
printf '%s\n' "${LIB}"
