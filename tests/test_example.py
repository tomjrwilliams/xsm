
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

@xt.nTuple.decorate()
class Handle_Incr(typing.NamedTuple):

    def matches(
        self,
        observer: xsm.observers.Handled,
        state: xsm.State
    ) -> bool:
        return True

    async def handle(
        self, states: xsm.States, broker: xsm.Broker
    ) -> Handle_Incr:
        for state in states:
            if isinstance(state, Prices):
                if state.curr.any(lambda v: v >= 1):
                    return self
                await state.update(
                    state.curr.map(lambda v: round(v + 0.1, 3)),
                    broker=broker
                )
        return self

@xt.nTuple.decorate()
class Handle_Print(typing.NamedTuple):

    def matches(
        self,
        observer: xsm.observers.Handled,
        state: xsm.State
    ) -> bool:
        return True

    async def handle(
        self, states: xsm.States, broker: xsm.Broker
    ) -> Handle_Print:
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
    
    obs_incr: xsm.observers.Simple = xsm.observers.Simple.with_handler(Handle_Incr())

    obs_print: xsm.observers.Simple =xsm.observers.Simple.with_handler(Handle_Print())

    observers = xsm.Observers([
        obs_incr,
        obs_print,
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
