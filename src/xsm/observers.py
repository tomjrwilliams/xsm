
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

    # NOTE: do not share mutable default value for queue
    # same issue as when doing so with kwargs

    @property
    def queue(self) -> collections.deque: ...

    @property
    def skip_empty(self) -> bool: ...

    @abc.abstractmethod
    def matches(self, state: xsm.State) -> bool: ...

    @abc.abstractmethod
    async def handle(
        self,
        states: xsm.States,
        broker: xsm.Broker,
    ) -> Simple: ...
        
    @staticmethod
    async def _flush(
        self,
        broker: xsm.Broker,
    ) -> Simple:
        states: xsm.States = xsm.States(self.queue)
        self.queue.clear()
        if self.skip_empty and not states.len():
            return self
        return await self.handle(states, broker)

    @staticmethod
    async def _receive(self: Simple, state: xsm.State[T]):
        if self.matches(state):
            self.queue.append(state)

    @staticmethod
    def interface():
        return dict(
            receive=Simple._receive,
            flush=Simple._flush,
        )

    @staticmethod
    def with_handler(
        handler: Handler,
        tags: xsm.Tags = xsm.Tags(),
        skip_empty: bool = True
    ):
        return Handled(
            handler,
            queue=collections.deque([]),
            tags=tags,
            skip_empty=skip_empty,
        )

# ------------------------------------------------------

class Handler(typing.Protocol):

    def matches(
        self: Handler, observer: Handled, state: xsm.State
    ) -> bool: ...

    async def handle(
        self, states: xsm.States, broker: xsm.Broker
    ) -> Handler: ...

# ------------------------------------------------------

@xt.nTuple.decorate(**Simple.interface())
class Handled(typing.NamedTuple):

    handler: Handler
    queue: collections.deque

    # NOTE: do not share mutable default values
    # same issue as when doing so with kwargs

    tags: xsm.Tags = xsm.Tags()

    skip_empty: bool = True
    
    async def receive( # type: ignore[empty-body]
        self, state: xsm.State
    ): ...

    async def flush( # type: ignore[empty-body]
        self, broker: xsm.Broker
    ) -> Handled: ...

    def matches(self, state: xsm.State) -> bool:
        return self.handler.matches(self, state)

    async def handle(
        self,
        states: xsm.States,
        broker: xsm.Broker,
    ) -> Handled:
        handler: Handler = await self.handler.handle(
            states, broker
        )
        return self._replace(
            queue=self.queue,
        )

# ------------------------------------------------------
