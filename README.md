# xsm

[![PyPI - Version](https://img.shields.io/pypi/v/xsm.svg)](https://pypi.org/project/xsm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/xsm.svg)](https://pypi.org/project/xsm)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install xsm
```

## Overview

xsm is a simple library for multiprocessing-powered state machines.

Each state in xsm should have fields for it's current and previous value, as well as a flag indicating if the state should be persisted.

```python
class State(typing.Protocol[T]):

    @property
    def curr(self) -> typing.Optional[T]: ...

    @property
    def prev(self) -> typing.Optional[T]: ...

    @property
    def persist(self) -> bool: ...

    ...
```

Every state creation / change is broadcast to all of the currently persisted states, as an event.

If an event matches a state, we then call for a handler: a function from the state and the event, to the new value of the state, and any other new states (which may or may not be persisted).

The handler, the state, and the event are then passed to a multi-processing pool for execution.

```python
class State(typing.Protocol[T]):

    ...

    @classmethod
    @abc.abstractmethod
    def dependencies(
        cls
    ) -> xt.iTuple[typing.Type[State]]: ...

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

```

Upon completion, the updated value of the state is assigned to the central state look-up table (unless current is set to None, in which case it is retired).

Whilst there are no guarantees on the order that events are applied to a given state, we do at least guarantee that only one operation can be in flight for a given state at a time (with the others added to a queue for later execution).

## License

`xsm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
