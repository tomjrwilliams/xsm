
from __future__ import annotations

import abc
import operator
import collections
import functools
import itertools

import inspect

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

T = typing.TypeVar('T')
V = typing.TypeVar('V')

# ------------------------------------------------------

def handle_prices(state: Prices, event: Prices) -> Prices:
    return state._replace(
        curr=state.curr.map(lambda v: v + 0.1),
        prev=state.curr,
    )

@xt.nTuple.decorate()
class Prices(typing.NamedTuple):
    """
    >>> states = xt.iTuple([Prices(xt.iTuple([0.0]))])
    >>> res = xsm.loop(states, iters = 10, timeout=5, processes=1)
    >>> res[0].curr.map(functools.partial(round, ndigits=3))
    iTuple(1.0)
    """

    curr: xt.Floats
    prev: typing.Optional[xt.Floats] = None

    persist: bool = True

    # --

    @classmethod
    def dependencies(cls):
        return xt.iTuple((
            Prices,
        ))
    
    def matches(self, event: xsm.Event) -> bool:
        return True

    def handler(self, event: xsm.Event):
        if isinstance(event, Prices):
            return handle_prices
        assert False, event

# ------------------------------------------------------

p: xsm.State = Prices(xt.Floats([1.]))