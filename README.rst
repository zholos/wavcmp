wavcmp
======

This program compares two audio files to determine if their waveforms closely
match, allowing for a relative time offset. The difference metric is the sum of
absolute differences between samples.

Run ``wavcmp.py`` from the source directory or install it (with optimizations if
Cython is available) using ``pip``::

    pip install [--user] .


Algorithm
---------

The naïve algorithm is to simply find the sum of absolute differences at each
offset and choose the offset that minimizes it. This is slow. Several
optimizations are applied progressively to obtain the same result in a
reasonable amount of time.

Only matches with a metric below a certain threshold (configured with the ``-t``
command-line option) are reported, so there is no need to keep summing absolute
differences for a particular offset after this threshold has been exceeded. This
is the "limit" in ``_limited_ds()``. Furthermore, as soon as a match is found
the threshold can be decreased relative to the metric of the match. This only
speeds up searches for tracks that do match, but is especially useful for tracks
that match at offset 0, which is checked first.

Since silence can be expected at the beginnings of most tracks, absolute
differences are calculated in pseudo-randomly ordered blocks. This is the
complex part of ``_limited_ds()``. Block size is chosen empirically. Smaller
blocks allow earlier detection that the threshold has been exceeded while larger
blocks increase throughput.

To reduce the amount of data being processed at each offset, a heuristic is
introduced to find match candidates more quickly in downsampled data. Using
``|x|+|y|>=|x+y|``::

    |a0-b0|+|a1-b1|+|a2-b2|+|a3-b3|>=|(a0+a1)-(b0+b1)|+|(a2+a3)-(b2+b3)|

The left-hand side is the sum of absolute differences over tracks *a* and *b*
with four samples, which is the match metric. The right-hand side is a lower
bound on it calculated using sums over groups of two samples each. If the lower
bound exceeds the threshold, the match candidate is rejected, otherwise it is
tested using the actual metric.

The sample groups must line up, so while the group sums for one of the tracks
are calculated once, the group sums for the other track are calculated for each
relative shift. This is the complex part of ``cmp_track()``. Group size is
chosen empirically. Smaller groups give a better lower bound while larger groups
mean less data to process.

After the group alignment is handled and the group sums are prepared, the actual
search involves a straightforward loop with integer variables and arrays. This
is encapsulated in ``_cmp_candidates()``. The same is true of ``_limited_ds()``.
Both these routines can be significantly optimized by using Cython to convert
them to C. Optimized versions of these routines are provided in the
``_compiled`` module and imported conditionally if Cython is available.
