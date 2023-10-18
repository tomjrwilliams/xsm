
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
    def prev(self) -> typing.Optional[T]: ...

    @abc.abstractmethod
    async def update(
        self: State[T], v_or_f: VorF[T], broker: Broker
    ) -> State[T]: ...

    @abc.abstractmethod
    def _replace(self: State[T], **kwargs) -> State[T]: ...

    @staticmethod
    def interface():
        return dict(update=update)

States = xt.iTuple[State[T]]

# ------------------------------------------------------

async def update(
    self: State[T],
    v_or_f: VorF[T],
    broker: Broker,
) -> State[T]:

    curr: T = self.curr

    if isinstance(v_or_f, type(curr)):
        v = typing.cast(T, v_or_f)
    else:
        v = typing.cast(typing.Callable[[T], T], v_or_f)(curr)

    res = self._replace(curr=v, prev=curr)
    await broker.receive(res)

    return res

# ------------------------------------------------------

class Observer(typing.Protocol):

    @property
    def tags(self) -> Tags: ...

    @abc.abstractmethod
    async def matches(self, state: State) -> bool: ...

    @abc.abstractmethod
    async def receive(self, state: State): ...

    @abc.abstractmethod
    async def flush(self, broker: Broker) -> Observer: ...

Observers = xt.iTuple[Observer]

# ------------------------------------------------------

async def flush(
    awaitable,
    f_done,
    start: datetime.datetime,
    seconds: typing.Optional[int] = None,
    timeout: bool = True,
):
    elapsed: float = (
        datetime.datetime.now() - start
    ).seconds

    if seconds is None:
        res = await awaitable
        done = f_done(res)
    elif elapsed >= seconds:
        res = None
        done = True
    elif timeout:
        try:
            res = await asyncio.wait_for(
                awaitable, 
                timeout = seconds - elapsed
            )
            done = f_done(res)
        except asyncio.TimeoutError:
            res = None
            done = True
    else:
        res = await awaitable
        done = f_done(res)

    return res, done

async def loop(
    broker: Broker,
    observers: Observers, 
    seconds: typing.Optional[int] = None,
    timeout: bool = True,
    iters: typing.Optional[int] = None
) -> Observers:

    done = False
    start = datetime.datetime.now()
    i = 0

    while not done:
        if iters is not None and i == iters:
            break
        _, done = await flush(
            broker.flush(observers),
            lambda changes: changes == 0,
            start,
            seconds=seconds,
            timeout=timeout,
        )
        if done: break
        _observers, done = await flush(
            asyncio.gather(*observers.map(
                operator.methodcaller("flush", broker)
            )),
            lambda _: done,
            start,
            seconds=seconds,
            timeout=timeout,
        )
        i += 1
        observers = xt.iTuple(_observers)

    return observers

# ------------------------------------------------------