
from __future__ import annotations

import enum

import operator
import collections
# import collections.abc
import functools
import itertools
from re import A
from tkinter import Y

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

import abc


# ------------------------------------------------------

E = typing.TypeVar('E', covariant=True)
T = typing.TypeVar('T')

# ------------------------------------------------------


class Tag(typing.NamedTuple):
    pass


# ------------------------------------------------------


class Event(typing.Protocol[E]):

    @property
    def tags(self) -> xt.iTuple[Tag]: ...

    @property
    def prev(self) -> E: ...

    @property
    def next(self) -> E: ...

    @abc.abstractmethod
    def __init__(
        self,
        tags: xt.iTuple[Tag],
        prev: E,
        next: E,
    ) -> None: ...

    # async?
    @abc.abstractmethod
    def dispatch(
        self: Event[E],
        #
    ): ...

def dispatch(self: Event[E]):
    return

# ------------------------------------------------------


class Observer(typing.Protocol):

    tags: xt.iTuple[Tag]

    @abc.abstractmethod
    def matches(
        self,
        event: Event,
    ):
        return


# ------------------------------------------------------


class State(typing.Protocol[T]):

    @property
    def tags(self) -> xt.iTuple[Tag]: ...

    @property
    def value(self) -> T: ...

    @abc.abstractmethod
    def _replace(
        self: State[T],
        *,
        tags: xt.iTuple[Tag],
        value: T,
    ) -> State[T]: ...

    @abc.abstractmethod
    def update(
        self: State[T],
        cls: typing.Type[Event[T]],
        value: T,
        #
    ) -> State[T]: ...


# ------------------------------------------------------


def update(
    self: State[T],
    cls: typing.Type[Event[T]],
    value: T,
):
    event: Event[T] = cls(self.tags, self.value, value)
    event.dispatch()
    return self._replace(tags = self.tags, value=value)


# ------------------------------------------------------

Floats = xt.iTuple[float]

# ------------------------------------------------------

@xt.nTuple.decorate(dispatch = dispatch)
class Prices_Event(typing.NamedTuple):

    tags: xt.iTuple[Tag]

    prev: Floats
    next: Floats
    
    def dispatch(  # type: ignore[empty-body]
        self: Prices_Event
    ): ...

# ------------------------------------------------------

@xt.nTuple.decorate(update = update)
class Prices(typing.NamedTuple):

    tags: xt.iTuple[Tag]
    value: Floats

    def update( # type: ignore[empty-body]
        self: State[Floats],
        cls: typing.Type[Event[Floats]],
        value: T,
    ) -> State[Floats]: ...

# ------------------------------------------------------

t: State = Prices(
    xt.iTuple([
        Tag(),
        #
    ]), 
    xt.iTuple([1.]),
    #
)

e: Event = Prices_Event(
    xt.iTuple([
        Tag(),
        #
    ]), 
    xt.iTuple([1.]),
    xt.iTuple([2.]),
)

t = t.update(
    Prices_Event,
    xt.iTuple([2.])
)

# ------------------------------------------------------
