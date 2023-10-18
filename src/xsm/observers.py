
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

class Simple(typing.Protocol):

    @property
    def tags(self) -> xsm.Tags: ...

    @property
    def queue(self) -> collections.deque: ...

    @abc.abstractmethod
    async def matches(self, state: xsm.State) -> bool: ...

    @abc.abstractmethod
    async def handle(self, states: xsm.States) -> Simple: ...
        
    @staticmethod
    async def _flush(self, broker: xsm.Broker) -> Simple:
        states: xsm.States = xsm.States(self.queue)
        self.queue.clear()
        return await self.handle(states, broker)

    @staticmethod
    async def _receive(self: Simple, state: xsm.State[T]):
        if (await self.matches(state)):
            self.queue.append(state)

    @staticmethod
    def interface():
        return dict(
            receive=Simple._receive,
            flush=Simple._flush
        )


# ------------------------------------------------------
