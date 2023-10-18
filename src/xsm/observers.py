
from __future__ import annotations

import abc
import operator
import collections
import functools
import itertools

import asyncio

import typing
import datetime

import numpy
import pandas # type: ignore

import jax # type: ignore
import jax.numpy # type: ignore
import jax.numpy.linalg # type: ignore

import jaxopt # type: ignore
import optax # type: ignore

import xtuples as xt

from . import xsm

# ------------------------------------------------------

T = typing.TypeVar('T')

# ------------------------------------------------------

class Asyncio_Deque(typing.Protocol):

    @property
    def tags(self) -> xsm.Tags: ...

    @property
    def queue(self) -> collections.deque: ...

    @abc.abstractmethod
    async def matches(self, state: xsm.State) -> bool: ...

    @staticmethod
    def interface():
        return dict(
            receive=receive,
        )
    
async def receive(self: Asyncio_Deque, state: xsm.State[T]):
    if (await self.matches(state)):
        self.queue.append(state)

# ------------------------------------------------------
