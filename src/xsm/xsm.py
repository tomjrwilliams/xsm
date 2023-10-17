
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

# ------------------------------------------------------

T = typing.TypeVar('T')

# ------------------------------------------------------


class Tag(typing.Protocol):
    pass


Tags = xt.iTuple[Tag]


# ------------------------------------------------------


class Broker(typing.Protocol):

    @abc.abstractmethod
    async def receive(self, state: State[T]):
        return

    @abc.abstractmethod
    async def flush(self):
        return

        
# ------------------------------------------------------

VorF = typing.Union[T, typing.Callable[[T], T]]

class State(typing.Protocol[T]):

    @property
    def curr(self) -> T: ...

    @property
    def tags(self) -> Tags: ...

    @property
    def broker(self) -> Broker: ...

    @property
    def prev(self) -> typing.Optional[T]: ...

    @abc.abstractmethod
    async def update(
        self: State[T], v_or_f: VorF[T]
    ) -> State[T]: ...

    @abc.abstractmethod
    def _replace(self: State[T], **kwargs) -> State[T]: ...

States = xt.iTuple[State[T]]

# ------------------------------------------------------

async def update(
    self: State[T],
    v_or_f: VorF[T],
) -> State[T]:

    curr: T = self.curr

    if isinstance(v_or_f, type(curr)):
        v = typing.cast(T, v_or_f)
    else:
        v = typing.cast(typing.Callable[[T], T], v_or_f)(curr)

    res = self._replace(curr=v, prev=curr)
    await self.broker.receive(res)

    return res

# ------------------------------------------------------

class Observer(typing.Protocol):

    @property
    def tags(self) -> Tags: ...

    @abc.abstractmethod
    async def receive(self, state: State): ...

Observers = xt.iTuple[Observer]

# ------------------------------------------------------

async def run(
    broker: Broker, 
    seconds: typing.Optional[int] = None,
    timeout: bool = True,
):
    done = False
    start = datetime.datetime.now()

    while not done:

        elapsed: float = (
            datetime.datetime.now() - start
        ).seconds

        if seconds is None:
            done = (await broker.flush()) == 0
        elif elapsed >= seconds:
            done = True
        elif timeout:
            try:
                done = (await asyncio.wait_for(
                    broker.flush(), timeout = seconds - elapsed
                )) == 0
            except asyncio.TimeoutError:
                done = True
        else:
            done = (await broker.flush()) == 0

# ------------------------------------------------------