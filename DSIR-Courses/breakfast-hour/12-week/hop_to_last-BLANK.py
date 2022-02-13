"""
Problem from GitHub user @vineetjohn
This problem was asked by Pinterest.

Given an integer list where each number represents the number of hops you can make, determine whether you can land on  the last index starting at index 0.

Additionally you may assume that you have to take the number of hops as the value of the index

For example, [2, 0, 1, 0] returns true while [1, 1, 0, 1] returns false.

>>> hop_to_last([2, 0, 1, 0])
True

>>> hop_to_last([1, 1, 0, 1])
False

>>> hop_to_last([2, 1])
False

"""
# Solution using recursion - https://www.python-course.eu/recursive_functions.php#:~:text=Definition%20of%20Recursion,this%20function%20a%20recursive%20function.
from tracemalloc import start


def reaches_last_helper(arr, start_index, target_index):
    if start_index == target_index:
        return True

    hop = arr[start_index]
    if not hop or (start_index + hop > target_index):
        return False

    return reaches_last_helper(arr, start_index + hop, target_index)

def hop_to_last(arr):
    return reaches_last_helper(arr, 0, len(arr)-1)


hop_to_last([2, 0, 1, 0])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
