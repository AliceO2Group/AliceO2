<!-- doxy
\page refFrameworkCorePROFILING Core PROFILING
/doxy -->

# Profiling using perf

## Setup 

the perf suite is a great profiler suite which is part of the Linux kernel itself. It provides very fine grained view of the state of a running system. By default, however the result you get on Ubuntu or Centos are not particulary good due to security reasons. However if the workstation you are running perf is your own, those defaults can probably be relaxed to provide a better profile expirience. First of all, allow profiling at all:

```bash
echo -1 > /proc/sys/kernel/perf_event_paranoid
```

then allow users to see the addres of kernel functions:

```bash
echo 0 > /proc/sys/kernel/kptr_restrict
```

In order to profile one needs to either run a command prefixing it with:

```
perf record -F 999 -g --call-graph dwarf
```

or the PID of the running process must be provided via

```
-p <PID>
```

alternatively you can profile all the processes on a given box using `-a`.

The above will accumulate results in a file called `perf.data`. Make sure you remove `perf.data` between runs, because they will otherwise accumulate.

In order to view the results, one must run `perf script`. e.g.:

```
perf script -i perf.data --no-inline | c++filt -r -t > profile.linux-perf.txt
```

results can then be visualised by drag and dropping them at <https://speedscope.app>.
