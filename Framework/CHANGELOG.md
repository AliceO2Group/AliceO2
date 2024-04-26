# 2024-03-14: Move DataProcessingDevice to use Signposts

All the messages from DataProcessingDevice have been migrated to use Signpost.
This will hopefully simplify debugging.

# 2024-02-22: Drop Tracy support

Tracy support never took off, so I am dropping it. This was mostly because people do not know about it and having a per process profile GUI was way unpractical. Moreover, needing an extra compile time flag meant one most likely did not have the support compiled in when needed.

I have therefore decided to replace it with signposts, which hopefully will see better adoption thanks
to the integration with Instruments on mac and the easy way they can be enabled dynamically.

We could then reintroduce Tracy support as a hook on top of signposts, if really needed.

# 2024-02-16: Improved Signposts.

In particular:

* New API so that Signposts can now act as a replacement of LOGF(info), LOGF(error), LOGF(warn).
* Improved documentation, including some hints about how to use `o2-log`.
* Bug fix to get `--signposts` work on a per device basis.

# 2024-01-10: Improved C++20 support.

Most of the macros which were failing when C++20 support is enabled now seem to work fine. The issue seems to be related to
some forward declaration logic which seems to be not working correctly in
ROOT 6.30.01. The issue is discussed in <https://github.com/root-project/root/issues/14230> and it seems to be not trivial to fix with the current ROOT version.
