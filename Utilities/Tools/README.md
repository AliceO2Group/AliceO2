# Using the Code checker / fixer

You can run the code checker and have it fix the errors it reports
(whenever possible) by doing:

    aliBuild init O2@dev                      # In case you have not a local checkout already
    O2_CHECKER_FIX=true alibuild build o2checkcode --defaults o2-daq --debug

At the end of which the sources in the O2 checkout directory should be
modified with the requested changes. You can modify which clang-tidy
checkers are used by setting the O2_CHECKER_CHECKS environment variable.

You can see what actually happens by looking at the `alidist/o2checkcode.sh`
recipe.
