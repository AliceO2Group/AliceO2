# Using the Code checker / fixer

You can run the code checker and have it fix the errors it reports
(whenever possible) by doing:

```bash
aliBuild init O2@dev                      # In case you have not a local checkout already
O2_CHECKER_FIX=true aliBuild build o2checkcode --defaults o2-daq --debug
```

At the end of which the sources in the O2 checkout directory should be
modified with the requested changes. You can modify which clang-tidy
checkers are used by setting the O2_CHECKER_CHECKS environment variable.

You can see what actually happens by looking at the `alidist/o2checkcode.sh`
recipe.

# Using the code checker on a restricted set of files

Often, only a few files are changed in the repository and running the codechecker
on the whole repository would be a considerable overhead. It is now possible to only check
files which were modified or are influenced by a modification by saying

```bash
ALIBUILD_BASE_HASH=commit_id aliBuild build o2checkcode --defaults o2-daq --debug
```

where `commit_id` is some git commit from which onwards we would like to check changed code.
Typically, `commit_id` should be the commit just before new modifications are applied.
Examples are `commit_id=HEAD` when we want to compare to the last git commit, 
or `commit_id=HEAD^^^` when we compare to the state 3 commits ago.

The pull request checker uses this mechanism to provide faster checks on github.

# Remove false positives from valgrind

[valgrind](http://valgrind.org) is a popular multiplatform memory
debugger and profiler. While it helps catching many subtle memory
error and leaks, it can be fooled by complex codebases like ROOT and
boost, reporting false positives which can limit its usefulness. For
this reason it offers the possibility to specify a suppression file
(via the `--suppressions=<filename>` option) to avoid reporting known
false positives. In case your false positives come from ROOT and
boost, you can use the `Utilities/Tools/boost-root.supp` file (notice
that different platforms and version of the software might require
adjustments in order to eliminate all the false positives).

# Configuring emacs to format according ALICE Formatting Guidelines

While not 100% ok, the file `Utilities/Tools/google-c-style.el` closely matches
our coding guidelines and can be used by emacs users as a template for their
environment.

# Code coverage

Code coverage reports for the AliceO2 repository are provideda thanks to the
courtesy of <https://codecov.io>.

This includes reports for changes in pull requests and for the commits on the
`dev` branch. PR reports will be posted automatically as part of the PR
discussion, while commits on the dev branch can be seen from the codecov gui:

https://codecov.io/gh/AliceO2Group/AliceO2

Notice that the system is smart enough to provide delta reports between the
base of a PR and its head.

The code coverage is done by compiling AliceO2 with the recently added
`-DCMAKE_BUILD_TYPE=COVERAGE` and by running the `ctest` suite afterwards. For
this reason, if you want to increase code coverage of your code, you should
make sure  you write a proper unit test for it (i.e. do not expect random
macros / executables sitting in random directories to affect the reports).
