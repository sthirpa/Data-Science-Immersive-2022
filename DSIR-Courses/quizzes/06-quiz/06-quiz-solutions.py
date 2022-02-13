"""
Quiz 6 - Sample solutions

Write a function that takes in a number n and returns `True` if there is a 13-snippet in the number.
If there is not a 13-snippet in the number, the function should return `False`.

Use this function to see if the following digit has a 13-snippit: 472917495027592

Rubric:
+1 : Correct structure of function (def)
+1 : Correct structure of function (params)
+1 : Correct structure of output (return, not a print)
+7 : Substantial progress towards correct answer (partial credit to grader's discretion)
+2 : Answer is correct
"""

# Method 1
# O(n) -- looks at each digit at most twice -- **best solution**
def snippet_13a(n, k=13):
    n = [int(digit) for digit in str(n).replace('0', '')]
    starti, endi = 0, 0         # start and end indexes of the sum so far
    curr_sum = 0                # running sum between starti and endi
    while endi < len(n):            # until we look at all digits ...
        if curr_sum == k:                 # found a k-snippet!
            return True
        elif curr_sum > k:                # if sum too large, remove starting digit
            curr_sum -= int(n[starti])
            starti += 1
        else:                             # if sum too small, add a new ending digit
            curr_sum += int(n[endi])
            endi += 1
    return False

snippet_13a(58528275927517572048726)
# Answer: True

# Method 2
# O(n^2) -- concise, but we do not break early from the inner loop once the sum > 13
def snippet_13b(n, k=13):
    n = [int(digit) for digit in str(n).replace('0', '')]
    return any(sum(n[i:i+j+1]) == k for i in range(len(n)) for j in range(len(n) - i))

snippet_13b(58528275927517572048726)
# Answer: True

# Method 3
# O(n) -- for each digit, looks at at most 13 other digits (not dependent on n)
def snippet_13c(n):
    n = [int(digit) for digit in str(n).replace('0', '')]
    for i in range(len(n)):
        for j in range(len(n) - i):
            curr_sum = sum(n[i:i+j+1])
            if curr_sum == 13: return True
            elif curr_sum > 13: break       # Ensures we only look at <= 13 digits. If removed, it is O(n^2)
    return False

snippet_13c(58528275927517572048726)
# Answer: True

snippet_13c(123456)
# Answer: False
