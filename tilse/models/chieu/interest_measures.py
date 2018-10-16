import numpy
from numba import jit


@jit(nopython=True)
def interest(similarities, date_diffs):
    """
    Computes interest as specified in Chieu and Lee (2004).

    The only differences is that we do not consider a reweighting of
    similarities by the "time span" of dates expressed in sentences (e.g.
    division by 30 if a sentence contains a reference to a month instead of a
    day), since we found the result of this operation to be insignificant,
    while it made the algorithm and computation more complex.

    Params:
        similarities (numpy.array): A matrix of sentence similarities.
        date_diffs (numpy.array): A matrix of date differences of sentences
        (e.g. difference between Sep 1 2001 and Sep 3 2001 is 2).

    Returns:
        A tuple consisting of a list of sentence ids with ranks (list(
        float)), and extents for the sentences (list(int)).
    """
    sentence_ranks = numpy.zeros(similarities.shape[0], dtype=numpy.uint16)
    extents = numpy.zeros(similarities.shape[0], dtype=numpy.uint16)

    interests = numpy.zeros((similarities.shape[0], 11), dtype=numpy.float32)

    for i in range(0, similarities.shape[0]):
        for j in range(0, similarities.shape[1]):
            for diff in range(1, 11):
                if date_diffs[i, j] <= diff:
                    interests[i][diff] += similarities[i, j]

        for diff in range(1, 11):
            if interests[i][diff] >= 0.8 * interests[i][10]:
                extents[i] = diff
                break

        sentence_ranks[i] = interests[i][10]

    return sentence_ranks, extents
