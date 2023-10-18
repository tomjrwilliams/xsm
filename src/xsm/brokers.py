
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

@xt.nTuple.decorate()
class Simple(typing.NamedTuple):

    queue: collections.deque = collections.deque()

    async def receive(self, state: xsm.State[T]):
        self.queue.append(state)

    async def flush(
        self, 
        observers: xsm.Observers,
    ) -> int:
        states = xsm.States(self.queue)
        self.queue.clear()

        await asyncio.gather(
            *states.map(lambda s: observers.filter(
                operator.methodcaller("matches", s)
            ).map(
                operator.methodcaller("receive", s)
            )).flatten()
        )
        return states.len()

# ------------------------------------------------------
