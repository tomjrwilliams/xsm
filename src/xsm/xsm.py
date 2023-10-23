
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
    async def receive(self, state: State[T]): ...

    @abc.abstractmethod
    async def flush(
        self,
        observers: Observers,
    ) -> int: ...

        
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

    # NOTE:
    # matches() should be a simple tags vs tags comparison
    # and thus doesn't need to be async (and should be natively parallelised within the map / filter - check?)
    
    # NOTE: receive(), however, may contain logic for eg.
    # making db / io calls to log the relevant state change
    # so does need to be async

    # NOTE: likewise, flush() may well have external calling logic
    # either to get data, or to store the event, other logging
    # so does need to be async

    # NOTE: because flush is called on every observer on every pass
    # it functions as a heart-beat as well as event response
    # at least assuming observers.Simple._flush(skip_empty=False)

    @property
    def tags(self) -> Tags: ...

    @abc.abstractmethod
    def matches(self, state: State) -> bool: ...

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

# TODO: remove the observers

# states themselves acc and flush?

# so then just loop broker flush

# state acc flush -> new states.flatten()


# one way is to build up a dict dynamically
# mapping state to matching events

# or vice versa, but then we have to reverse
# to acc all in at once?

# unless we acc concurrently

# i suppose if we can call for event in broker:
# generator

# that draws from a continuously re-built back up queue

# always append to the end and draw from the front

# the only issue is the lack of parallelisation


# the events can pile up if we're not batch handlnig indpendent events

# but if we have an explodign event queue that will always be an isue?


# i'd say then

# both batch

# but cut the batch and restart

# at each collision, where the same 

# state is triggered again in the batch


# and or have some kind of secondary queue

# or some notion of a reduce statement

# for combined updates tro the state

# including just implementing as a fold

# on those within the batch

# NOTE:

# if they returned the events

# as well as the state

# or if the state is just the new value

# together with the tags


# then there's no need to worry about a central querue

# can be just re-agg here


# and so can presumably be multi thread executed


# re-agg and fed back through


# so no need to pass the broker


# just return [state] [event]

# or even just [state] and imply the events from the state

# state, [state]
# from left infer any changes to fire

# right are assumed new

# if left is none then retire

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
        _observers, done = await flush(
            asyncio.gather(*observers.map(
                operator.methodcaller("flush", broker)
            )),
            lambda _: done,
            start,
            seconds=seconds,
            timeout=timeout,
        )
        observers = xt.iTuple(_observers)
        _, done = await flush(
            broker.flush(observers),
            lambda changes: changes == 0,
            start,
            seconds=seconds,
            timeout=timeout,
        )
        i += 1

    return observers

# ------------------------------------------------------
