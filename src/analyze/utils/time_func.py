import time
from collections.abc import Callable
from typing import Any


def time_func(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """
    Measures the execution time of a given function and returns both the result and the execution time.

    Parameters:
    - func (Callable[..., Any]): The function whose execution time is to be measured.
    - *args (Any): Positional arguments to be passed to the function.
    - **kwargs (Any): Keyword arguments to be passed to the function.

    Returns:
    - tuple[Any, float]: A tuple containing the function's result and the time taken to execute it in seconds.
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    return result, execution_time
