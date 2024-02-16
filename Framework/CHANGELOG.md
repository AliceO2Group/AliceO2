# 2024-02-16: Improved Signposts.

In particular:

* New API so that Signposts can now act as a replacement of LOGF(info), LOGF(error), LOGF(warn).
* Improved documentation, including some hints about how to use `o2-log`.
* Bug fix to get `--signposts` work on a per device basis.

# 2024-01-10: Improved C++20 support.

Most of the macros which were failing when C++20 support is enabled now seem to work fine. The issue seems to be related to
some forward declaration logic which seems to be not working correctly in
ROOT 6.30.01. The issue is discussed in <https://github.com/root-project/root/issues/14230> and it seems to be not trivial to fix with the current ROOT version.
