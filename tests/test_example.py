
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
import xsm

# ------------------------------------------------------

@xt.nTuple.decorate(**xsm.State.interface())
class Prices(typing.NamedTuple):

    tags: xsm.Tags

    # --

    curr: xt.Floats
    prev: typing.Optional[xt.Floats] = None

    # --
    
    async def update( # type: ignore[empty-body]
        self: Prices,
        v_or_f: xsm.VorF[xt.Floats],
        broker: xsm.Broker
    ) -> Prices: ...

# ------------------------------------------------------

@xt.nTuple.decorate(**xsm.observers.Simple.interface())
class Observer_Incr(typing.NamedTuple):

    tags: xsm.Tags = xsm.Tags()
    queue: collections.deque = collections.deque([])
    
    #  --

    async def receive( # type: ignore[empty-body]
        self, state: xsm.State
    ): ...

    async def flush( # type: ignore[empty-body]
        self, broker: xsm.Broker
    ) -> Observer_Incr: ...
    
    #  --

    def matches(self, state: xsm.State):
        return True

    async def handle(
        self, states: xsm.States, broker: xsm.Broker
    ) -> Observer_Incr:
        for state in states:
            if isinstance(state, Prices):
                if state.curr.any(lambda v: v >= 1):
                    return self
                await state.update(
                    state.curr.map(lambda v: round(v + 0.1, 3)),
                    broker=broker
                )
        return self

@xt.nTuple.decorate(**xsm.observers.Simple.interface())
class Observer_Print(typing.NamedTuple):

    tags: xsm.Tags = xsm.Tags()
    queue: collections.deque = collections.deque([])
    
    #  --

    async def receive( # type: ignore[empty-body]
        self, state: xsm.State
    ): ...

    async def flush( # type: ignore[empty-body]
        self, broker: xsm.Broker
    ) -> Observer_Incr: ...
    
    #  --

    def matches(self, state: xsm.State):
        return True

    async def handle(
        self, states: xsm.States, broker: xsm.Broker
    ) -> Observer_Print:
        for state in states:
            print(state.curr)
        return self
        
# ------------------------------------------------------

async def main_example():
    """
    >>> asyncio.run(main_example())
    iTuple(0.0)
    iTuple(0.1)
    iTuple(0.2)
    iTuple(0.3)
    iTuple(0.4)
    iTuple(0.5)
    iTuple(0.6)
    iTuple(0.7)
    iTuple(0.8)
    iTuple(0.9)
    iTuple(1.0)
    """
    broker: xsm.Broker = xsm.brokers.Simple()
    observers = xsm.Observers([
        Observer_Incr(),
        Observer_Print(),
    ])

    prices: Prices = Prices(
        xsm.Tags(), xt.Floats([0.]),
    )
    prices = await prices.update(xt.Floats([.0]), broker)

    await xsm.loop(broker, observers, seconds = 5)

# ------------------------------------------------------

# class Asset:
# tag: name? id?
# state: value

# class Position:
# tag: ? asset
# state: quantity
# state: value (?)

# class Strategy:
# ...: positions
# 


# todo: make the tag an actual property
# of individual fields
# and tag as union of eg
# name, id, etc. 
# with user type as a final protocol (?)

# ------------------------------------------------------
