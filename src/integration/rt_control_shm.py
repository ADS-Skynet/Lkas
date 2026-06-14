"""Minimal POSIX SHM writer for LKAS -> DCAS raw control samples."""

from __future__ import annotations

import ctypes
import mmap
import os
import time
from typing import Optional

SHM_NAME = "/rt_control_shm"
SHM_PATH = "/dev/shm/rt_control_shm"
MAGIC = 0x5243544C  # 'RCTL'
VERSION = 2
RING_CAPACITY = 64


class LkasToDcasSample(ctypes.Structure):
    _fields_ = [
        ("timestamp_us", ctypes.c_uint64),
        ("lkas_throttle", ctypes.c_float),
        ("lkas_steering", ctypes.c_float),
        ("reserved", ctypes.c_uint32),
    ]


class ActuatorToDcasSample(ctypes.Structure):
    _fields_ = [
        ("timestamp_us", ctypes.c_uint64),
        ("current_speed_kmh", ctypes.c_float),
        ("hardware_fault", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class DcasToActuatorSample(ctypes.Structure):
    _fields_ = [
        ("timestamp_us", ctypes.c_uint64),
        ("final_throttle", ctypes.c_float),
        ("is_valid", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class RingBufferLkas(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("entries", LkasToDcasSample * RING_CAPACITY),
    ]


class RingBufferActuator(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("entries", ActuatorToDcasSample * RING_CAPACITY),
    ]


class RingBufferDcas(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("entries", DcasToActuatorSample * RING_CAPACITY),
    ]


class RtControlShmLayout(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("ring_capacity", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("actuator_to_dcas", RingBufferActuator),
        ("lkas_to_dcas", RingBufferLkas),
        ("dcas_to_actuator", RingBufferDcas),
    ]


class RtControlShmWriter:
    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._mmap: Optional[mmap.mmap] = None
        self._layout: Optional[RtControlShmLayout] = None
        self._open()

    def _open(self) -> None:
        if not os.path.exists(SHM_PATH):
            return

        self._fd = os.open(SHM_PATH, os.O_RDWR)
        size = ctypes.sizeof(RtControlShmLayout)
        self._mmap = mmap.mmap(self._fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self._layout = RtControlShmLayout.from_buffer(self._mmap)

        if self._layout.magic != MAGIC or self._layout.version != VERSION:
            self.close()

    def is_ready(self) -> bool:
        return self._layout is not None

    def write_lkas_to_dcas(self, throttle: float, steering: float) -> bool:
        if not self._layout:
            return False

        ring = self._layout.lkas_to_dcas
        head = ring.head
        idx = head % RING_CAPACITY

        sample = LkasToDcasSample()
        sample.timestamp_us = time.time_ns() // 1000
        sample.lkas_throttle = float(throttle)
        sample.lkas_steering = float(steering)
        sample.reserved = 0

        ring.entries[idx] = sample
        ring.head = head + 1
        return True

    def close(self) -> None:
        if self._layout is not None:
            # Drop ctypes buffer views before closing mmap
            self._layout = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._layout = None
