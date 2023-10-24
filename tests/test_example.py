
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
    
    def matches(self, state: xsm.State) -> bool:
        return True
    
    @staticmethod
    def handle_prices(
        state: Prices, event: Prices
    ):
        return state._replace(
            curr=state.curr.map(lambda v: v + 0.1),
            prev=state.curr,
        )

    def handler(self, state: xsm.State[V]) -> typing.Callable[
        [Prices, xsm.State[V]], xsm.Res
    ]:
        if isinstance(state, Prices):
            return self.handle_prices
        assert False, state

# ------------------------------------------------------
