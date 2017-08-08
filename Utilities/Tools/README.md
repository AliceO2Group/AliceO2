# Using the Code checker / fixer

You can run the code checker and have it fix the errors it reports
(whenever possible) by doing:

    aliBuild init O2@dev                      # In case you have not a local checkout already
    O2_CHECKER_FIX=true aliBuild build o2checkcode --defaults o2-daq --debug

At the end of which the sources in the O2 checkout directory should be
modified with the requested changes. You can modify which clang-tidy
checkers are used by setting the O2_CHECKER_CHECKS environment variable.

You can see what actually happens by looking at the `alidist/o2checkcode.sh`
recipe.

# Using the code checker on a restricted set of files

Often, only a few files are changed in the repository and running the codechecker
on the whole repository would be a considerable overhead. It is now possible to only check
files which were modified or are influenced by a modification by saying

```
ALIBUILD_BASE_HASH=commit_id aliBuild build o2checkcode --defaults o2-daq --debug
```

where `commit_id` is some git commit from which onwards we would like to check changed code.
Typically, `commit_id` should be the commit just before new modifications are applied.
Examples are `commit_id=HEAD` when we want to compare to the last git commit, 
or `commit_id=HEAD^^^` when we compare to the state 3 commits ago.

The pull request checker uses this mechanism to provide faster checks on github.

# Configuring emacs to format according ALICE Coding guidelines

While not 100% ok, the file `Utilities/Tools/google-c-style.el` closely matches
our coding guidelines and can be used by emacs users as a template for their
environment.
