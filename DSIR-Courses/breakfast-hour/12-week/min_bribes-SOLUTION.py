"""
Source: HackerRank - https://www.hackerrank.com/dashboard

It is New Year's Day and people are in line for the Wonderland rollercoaster ride. Each person wears a sticker indicating their initial position in the queue. Initial positions increment by 1 from 1 at the front of the line to n at the back.

Any person in the queue can bribe the person directly in front of them to swap positions. If two people swap positions, they still wear the same sticker denoting their original places in line. One person can bribe at most two others. For example, if n=8 and Person 5 bribes Person 4, the queue will look like this: [1,2,3,5,4,6,7,8].

Fascinated by this chaotic queue, you decide you must know the minimum number of bribes that took place to get the queue into its current state. If anyone has bribed more than two people, the line is too chaotic to compute the answer.

Return the minimum number of bribes necessary or 'Too chaotic' if someone has bribed more than 2 people.
Input Format

>>> min_bribes([2, 1, 5, 3, 4])
3

>>> min_bribes([2, 5, 1, 3, 4])
'Too chaotic'

>>> min_bribes([5, 1, 2, 3, 7, 8, 6, 4])
'Too chaotic'

>>> min_bribes([1, 2, 5, 3, 7, 8, 6, 4])
7

"""

def min_bribes(q):
    count = 0
    for index, value in enumerate(q):
        if index+1 < value and abs(index +1 - value) > 2:
            return "Too chaotic"
        for j in range(index+1, len(q)):
            if value > q[j]:
                count += 1
    return count
    
# Code by John
def min_bribes(q):
    bribes = [sorted(q).index(person)-q.index(person) for person in sorted(q)]
    pos_bribes = [bribe for bribe in bribes if bribe > 0]
    neg_bribes = [bribe for bribe in bribes if bribe < 0]

    adjustment = 0
    for neg_bribe in neg_bribes:
        if neg_bribes > -2:
            adjustment += 1
    for pos_bribe in pos_bribes:
        if pos_bribe > 2:
            return "Too chaotic"

    return sum(pos_bribes) + adjustment    


min_bribes([2, 1, 5, 3, 4])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
