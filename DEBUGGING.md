# How do I run in debug mode?

By default, O2 builds with optimizations (`-O2`) turned on, while leaving debug symbols available.
This allows doing some simple set of debugging operations, e.g. getting a reasonable stacktrace, however it does not work when using a debugger. This results in the typical effect of single stepping in gdb / lldb jumping around the sourcecode.
In order to fix this you need to turn off the optimization and there are several ways you could do this, depending on how permanent you want the change to be.

* Add `-DCMAKE_BUILD_TYPE=Debug` in `alidist/o2.sh` and then rebuild using aliBuild. This will be a permanent change, however it requires rebuilding everything.
* Change `sw/BUILD/<architecture>/O2-latest/O2/CMakeCache.txt` to have `CMAKE_BUILD_TYPE=Debug` and the type `ninja install` in the same folder. This will be undone by the next time you run aliBuild, however it has the advantage that the ninja command can be targeted to a specific subsystem, e.g. `ninja Framework/install`. Notice also that by default `-Og` is used, so you might have to change `CMAKE_CXX_FLAGS_DEBUG` in the same file to use `-O0`.
* Change `sw/BUILD/<architecture>/O2-latest/O2/build.ninja` to use `-Og` or `-Os`, for the specific targets you are interested in. This in the most fine grained option.
