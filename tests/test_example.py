
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

    # TODO: replace with separate tag fields
    # on the named tuple

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

    # TODO: needs matches, flush

    # return from the flush as a list
    # which is flattened back down for the loop

    # ie so can spawn new states, and retire itself by returning empty 

# ------------------------------------------------------

# TODO:

# return:

# self state, two tuple
# of either old, new or old none if retire

# and then iterable of tuples

# one tuple is event, no state

# two tuple is a new state, presume prev is none


# on match, generate a task

# that will on run, call the relevant state handler returning the above


# where the task should be given a callback

# that updates the registry of states with the response for self

# as well as cahcing the update and events into the qeue for processing


# but that means we can't concecutively store up tasks

# only one can be in flight per state at a given time

# as once the task is created, the state is locked in


# so the callback also needs ot handle that lock rtemoval

# eg. just a set of currently in flight states

# that we skip over from the qeue until done


# https://docs.python.org/3.9/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor

# futures.as_completed

# iterator of results

# use the above for callbacks

# once to do is less than a certain offset from workers

# or timeout (?)

# ie. certain number presumably idle

# then collect up those not done and re batch

# eg. then call wait on the rest with a tiny callback to filter into done and not done (if not all done)

# and then go back to the queue to add to the remaining not done


# so then we can have a central state registry keeping track of both what to schedule

# including what states are already in flight

# and what values each of the states have


# so similar to the above, presumably if match

# return a future (versus a task before)
# that can be sent for execution


# so the states need to be given the executor

# when we know there's a match, for them to call executor.submit(...)


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
    broker: xsm.Broker = xsm.brokers.Simple(
        collections.deque()
    )
    
    obs_incr = xsm.observers.Simple.with_handler(Handle_Incr())
    obs_print = xsm.observers.Simple.with_handler(Handle_Print())

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
