
from __future__ import annotations

import abc
import operator
import collections
import functools
import itertools

import asyncio
import concurrent.futures as conc_fut

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
V = typing.TypeVar('V')

# ------------------------------------------------------

class State(typing.Protocol[T]):

    @property
    def curr(self) -> typing.Optional[T]: ...

    @property
    def prev(self) -> typing.Optional[T]: ...

    @property
    def persists(self) -> bool: ...

    @classmethod
    @abc.abstractmethod
    def dependencies(
        cls
    ) -> xt.iTuple[typing.Type[State]]: ...
    # @functools.lru_cache(maxsize=1)

    @abc.abstractmethod
    def matches(
        self, state: State[V] #
    ) -> bool: ...

    @abc.abstractmethod
    def handler(
        self, state: State[V] #
    ) -> typing.Callable[
        [State[T], State[V]], Res
    ]: ...

# ------------------------------------------------------

States = xt.iTuple[State[T]]

iState = tuple[int, State]

Res = tuple[State, States]
Res_Fut = conc_fut.Future[Res]

Handler = typing.Callable[
    [State[T], State[V]], Res
]

Handler_Event = tuple[Handler, State]

State_Queue = collections.deque[State]
Event_Queue = dict[
    int, collections.deque[Handler_Event]
]

# ------------------------------------------------------

def match_event(
    e: State,
    states: dict[int, State],
):
    t = type(e)
    for i, s in states.items():
        # for cls in depends_on[t]:
            # for i in by_cls[cls]:
                # s = states[i]
                # ...
        if t in s.dependencies():
            if s.matches(e):
                yield i, s.handler(e)

def f_submit(
    states: dict[int, State],
    s_queue: State_Queue,
    s_pending: set[int],
) -> typing.Callable[
    [
        int,
        typing.Callable,
        State,
        conc_fut.Executor
    ], Res_Fut
]:
    
    id = len(states)

    def i_callback(i: int):
        def callback(res: Res_Fut):
            nonlocal id
            
            state, others = res.result()

            if state.curr is None:
                del states[i]
            else:
                states[i] = state.curr
            for s in others:
                if s.persists:
                    id += 1
                    states[id] = s.curr

            s_queue.append(state)
            s_queue.extend(others)

            return
        return callback

    def f(i, h, e, executor: conc_fut.Executor):
        s = states[i]
        s_pending.add(i)
        fut = executor.submit(h, s, e)
        fut.add_done_callback(i_callback(i))
        return fut
    return f

# ------------------------------------------------------

def loop(
    init: States,
    iters: typing.Optional[int] = None,
    timeout: typing.Optional[float] = None,
    max_workers: typing.Optional[int] = None,
) -> States:

    start = datetime.datetime.now()

    states: dict[int, State] = dict(
        init.filter(lambda s: s.persists)
        .enumerate()
        #
    )

    s_queue: State_Queue = collections.deque(init)
    e_queue: Event_Queue = dict()

    f_pending: set[Res_Fut] = set()
    s_pending: set[int] = set()
    
    submit = f_submit(states, s_queue, s_pending)

    i: int
    e: State
    h: Handler

    n_done: int = 0

    with conc_fut.ProcessPoolExecutor(
        max_workers=max_workers,
        # 
    ) as executor:
        n_workers: int = len(executor._processes)

        done: bool = False

        while not done:

            while len(f_pending) >= n_workers:
                f_done, f_pending = conc_fut.wait(
                    f_pending,
                    timeout = (
                        None if timeout is None
                        else (
                            datetime.datetime.now() - start
                        ).total_seconds()
                    ),
                    return_when=conc_fut.FIRST_COMPLETED,
                )
                n_done += len(f_done)

            if len(e_queue):
                e_pop = set()
                for i, es in e_queue.items():
                    if i in s_pending:
                        continue

                    h, e = es.popleft()
                    if not len(es):
                        e_pop.add(i)

                    f_pending.add(submit(
                        i, h, e, executor
                    ))

                for i in e_pop:
                    e_queue.pop(i)

            if len(s_queue):
                e = s_queue.popleft()

                for i, h in match_event(
                    e, states
                ):
                    if not i in s_pending:
                        f_pending.add(submit(
                            i, h, e, executor
                        ))
                        continue

                    if i not in e_queue:
                        e_queue[i] = collections.deque([])
                    e_queue[i].append((h, e,))
            
            if timeout is not None:
                done = done or (
                    datetime.datetime.now() - start
                ).total_seconds() > timeout
            
            if iters is not None:
                done = done or n_done > iters
            
            if (
                not len(f_pending)
                and not len(s_queue)
                and not len(e_queue)
            ):
                done = True
            
    return States(states.values())

# ------------------------------------------------------
