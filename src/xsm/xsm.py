
from __future__ import annotations

import abc
import operator
import collections
import functools
import itertools

import asyncio

import multiprocessing as mp
import multiprocessing.pool as mp_pool

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
U = typing.TypeVar('U', covariant=True)
V = typing.TypeVar('V')

# ------------------------------------------------------

class Message(typing.Protocol[U]):

    @property
    def curr(self) -> typing.Optional[U]: ...

    @property
    def prev(self) -> typing.Optional[U]: ...

    @property
    def persist(self) -> bool: ...

# ------------------------------------------------------

class State(typing.Protocol[T]):

    @property
    def curr(self) -> typing.Optional[T]: ...

    @property
    def prev(self) -> typing.Optional[T]: ...

    @property
    def persist(self) -> bool: ...

    @classmethod
    @abc.abstractmethod
    def dependencies(cls) -> xt.iTuple[typing.Type[State]]: ...

    @abc.abstractmethod
    def matches(
        self: State[T], 
        event: typing.Union[Message[V], State[U]]
    ) -> bool: ...

    @abc.abstractmethod
    def handler(
        self: State[T], 
        event: typing.Union[Message[V], State[U]] #
    ) -> typing.Callable[
        [State[T], typing.Union[Message[V], State[U]]], Res
    ]: ...

# ------------------------------------------------------

Event = typing.Union[Message, State]
Events = xt.iTuple[Event]

States = xt.iTuple[State]

iState = tuple[int, State]

Res = typing.Union[State, tuple[State, Events]]
Res_Async = mp_pool.AsyncResult[Res]

Handler = typing.Callable[
    [State, Event], Res
]

Handler_Event = tuple[Handler, Event]

State_Queue = collections.deque[Event]
Event_Queue = dict[
    int, collections.deque[Handler_Event]
]

# ------------------------------------------------------

def match_event(
    e: Event,
    states: dict[int, State],
    depends_on,
    triggers,
):
    t = type(e)
    for i, s in states.items():

        if t not in triggers:
            triggers[t] = set()

        if type(s) in triggers[t]:
            if s.matches(e):
                yield i, s.handler(e)

def f_submit(
    states: dict[int, State],
    s_queue: State_Queue,
    s_pending: set[int],
    depends_on,
    triggers,
) -> typing.Callable[
    [
        int,
        typing.Callable,
        Event,
        mp_pool.Pool
    ], Res_Async
]:
    
    id = len(states)

    def i_callback(i: int):
        def callback(result: Res):
            nonlocal id
            nonlocal i

            state: Event
            others: Events

            # result: Res = res.result()
            
            # NOTE: below is because we can return a single state rather than an iterable so have to disambiguate, because states themselves as generally namedtuples are themselves iterable
            if (
                hasattr(result, "curr")
                and hasattr(result, "prev")
                and hasattr(result, "persist")
            ):
                state = typing.cast(State, result)
                others = xt.iTuple()
            else:
                state, others = result

            if state.curr is None:
                del states[i]
            elif state.persist:
                states[i] = typing.cast(State, state)

            for _i, s in enumerate((state,) + others):
                t = type(s)
            
                if t not in depends_on:

                    depends_on[t] = set(s.dependencies())
                    
                    for dep in depends_on[t]:
                        if dep not in triggers:
                            triggers[dep] = set()
                        triggers[dep].add(t)

                if _i > 0 and s.persist:
                    id += 1
                    states[id] = s

                s_pending.remove(i)

            s_queue.append(state)
            s_queue.extend(others)

            return
        return callback

    # def f(i, h, e, executor: conc_fut.Executor):

    def f(i, h, e, pool):
        s = states[i]
        s_pending.add(i)

        # fut = executor.submit(h, s, e)
        # fut.add_done_callback(i_callback(i))

        fut = pool.apply_async(
            h,
            (s, e,),
            callback=i_callback(i),
        )

        return fut
    return f

# ------------------------------------------------------

def loop(
    init: Events,
    *,
    iters: typing.Optional[int] = None,
    timeout: typing.Optional[float] = None,
    processes: typing.Optional[int] = None,
) -> States:

    start = datetime.datetime.now()

    states: dict[int, State] = dict(
        init.filter(lambda s: s.persist)
        .map(lambda s: typing.cast(State, s))
        .enumerate()
        #
    )

    s_queue: State_Queue = collections.deque(init)
    e_queue: Event_Queue = dict()

    f_pending: set[Res_Async] = set()
    s_pending: set[int] = set()

    depends_on: dict[
        typing.Type[State], set[typing.Type[State]]
    ] = {}
    triggers: dict[
        typing.Type[State], set[typing.Type[State]]
    ] = {}

    for s in init:
        t = type(s)
        if t not in depends_on:
            depends_on[t] = set(s.dependencies())

    for t, deps in depends_on.items():
        for dep in deps:
            if dep not in triggers:
                triggers[dep] = set()
            triggers[dep].add(t)

    submit = f_submit(
        states,
        s_queue,
        s_pending,
        depends_on,
        triggers,
    )

    i: int
    e: Event
    h: Handler
    
    # with conc_fut.ProcessPoolExecutor(
    #     max_workers=max_workers,
    #     # 
    # ) as executor:

    with mp.Pool(
        processes=processes,
    ) as pool:

        n_workers: int
        # n_workers: int = len(executor._processes)

        if processes is None:
            n_workers = mp.cpu_count()
        else:
            n_workers = processes

        done: bool = False
        it: int = 0

        while not done:

            while len(f_pending) >= n_workers:
                f_pending = set(
                    f for f in f_pending
                    if not f.ready()
                )
                # f_done, f_pending = conc_fut.wait(
                #     f_pending,
                #     timeout = (
                #         None if timeout is None
                #         else (
                #             datetime.datetime.now() - start
                #         ).total_seconds()
                #     ),
                #     return_when=conc_fut.FIRST_COMPLETED,
                # )
                # n_done += len(f_done)

            if len(e_queue):
                e_pop = set()
                for i, es in e_queue.items():
                    if i in s_pending:
                        continue

                    h, e = es.popleft()
                    if not len(es):
                        e_pop.add(i)

                    f_pending.add(submit(
                        i, h, e, pool
                    ))

                for i in e_pop:
                    e_queue.pop(i)

            if len(s_queue):
                e = s_queue.popleft()

                for i, h in match_event(
                    e, states, depends_on, triggers
                ):
                    if not i in s_pending:
                        f_pending.add(submit(
                            i, h, e, pool
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
                done = done or it >= iters
            
            if (
                not len(f_pending)
                and not len(s_queue)
                and not len(e_queue)
            ):
                done = True
            
            it += 1
            
    return States(states.values())

# ------------------------------------------------------
