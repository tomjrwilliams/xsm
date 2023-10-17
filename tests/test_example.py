
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

@xt.nTuple.decorate(update=xsm.update)
class Prices(typing.NamedTuple):

    broker: xsm.Broker
    tags: xsm.Tags

    # --

    curr: xt.Floats
    prev: typing.Optional[xt.Floats] = None

    # --
    
    async def update( # type: ignore[empty-body]
        self: Prices, v_or_f: xsm.VorF[xt.Floats]
    ) -> Prices: ...

# ------------------------------------------------------

@xt.nTuple.decorate()
class Observer_Example(typing.NamedTuple):

    tags: xsm.Tags = xsm.Tags()

    async def receive(self, state: xsm.State):
        print("receive")
        print(state.curr)

        if isinstance(state, Prices):
            await asyncio.sleep(.5)
            await state.update(xt.Floats([.3]))

# ------------------------------------------------------

def test_example():

    async def main():
        print("--")
        broker: xsm.Broker = xsm.brokers.Asyncio(
            xt.iTuple([Observer_Example()])
        )
        prices: Prices = Prices(
            broker, xsm.Tags(), xt.Floats([1.]),
        )
        prices = await prices.update(xt.Floats([.2]))

        return await xsm.run(broker, seconds = 5)

    asyncio.run(main())

# ------------------------------------------------------
