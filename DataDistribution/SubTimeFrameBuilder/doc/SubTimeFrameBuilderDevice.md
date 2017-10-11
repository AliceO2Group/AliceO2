% SubTimeFrameBuilderDevice(1)
% Gvozden Nešković <gvozden.neskovic@cern.ch>
% September 2018

# NAME

SubTimeFrameBuilderDevice – aggregate readout data into Sub Time Frames (STF)


# DESCRIPTION

**SubTimeFrameBuilderDevice** (StfBuilder) is the O2 device responsible for aggregating readout data into
SubTimeFrame objects on the FLPs. On the input channel, the StfBuilder  receives HBFrame data
from the readout. On the output, the StfBuilder supports the DPL or the native data distribution
chain forwarding (to SubTimeFrameSender). Optionally, the StfBuilder can be used to write STFs
to files, before they are sent out.

# OPTIONS

## General options

**-h**, **--help**
:   Print help

**-v** **--version**
:   Print version

**--severity** level
:   Log severity level: trace, debug, info, state, warn, error, fatal, nolog.
    The default value of this parameter is '*debug*'.

**--verbosity** level
:   Log verbosity level: veryhigh, high, medium, low.
    The default value of this parameter is '*medium*'.

**--color** arg
:   Log color (true/false). The default value of this parameter is '*1*'.

**--log-to-file** filename
:   Log output to a file.

**--print-options** [arg]
:   Print options.  The default value of this parameter is '*1*'.


## FairMQ device options

**--id** arg
:   Device ID (**required**).

**--io-threads** n
:   Number of I/O threads. The default value of this parameter is '*1*'.

**--transport** arg (=zeromq)
:   Transport ('zeromq'/'nanomsg'/'shmem'). The default value of this parameter is '*zeromq*'.

**--network-interface** arg
:   Network interface to bind on (e.g. eth0, ib0..., default will try to detect the interface of
    the default route).
    The default value of this parameter is '*default*'.

**--session** arg
:   Session name. All FairMQ devices in the chain must use the same session parameter.
    The default value of this parameter is '*default*'.


## FairMQ channel parser options

**--mq-config** path
:   JSON input as file.

**--channel-config** conf
:   Configuration of single or multiple channel(s) by comma separated *key=value* list.


## SubTimeFrameBuilderDevice options

**--input-channel-name** name
:   Name of the input readout channel (**required**).

**--stand-alone**
:   Standalone operation. SubTimeFrames will not be forwarded to other processes.

**--max-buffered-stfs** num
:   Maximum number of buffered SubTimeFrames before starting to drop data. Unlimited: -1.
    The default value of this parameter is '*-1*'.

**--output-channel-name** name
:   Name of the output channel for non-DPL deployments (**required**).

**--cru-count** num
:   Number of CRU Readout processes (each connects to the readout channel, index 0 to N-1).
    The default value of this parameter is '*1*'.

**--gui**
:   Enable GUI.

## SubTimeFrameBuilderDevice DPL options

**--enable-dpl**
:   Enable DPL.

**--dpl-channel-name** name
:   Name of the DPL output channel for DPL deployments (**required** when --enable-dpl is used).


## (Sub)TimeFrame file sink options

**--data-sink-enable**
:   Enable writing of (Sub)TimeFrames to file.

**--data-sink-dir** dir
:   Specifies a root directory where (Sub)TimeFrames are to be written.
    Note: A new directory will be created here for all files of the current run.

**--data-sink-file-name** pattern
:   Specifies file name pattern: %n - file index, %D - date, %T - time.
    The default value of this parameter is '*%n*'.

**--data-sink-max-stfs-per-file** num
:   Limit the number of (Sub)TimeFrames per file.
    The default value of this parameter is '*1*'.

**--data-sink-max-file-size** arg (=4294967296)
:   Limit the target size for (Sub)TimeFrame files.'
    Note: Actual file size might exceed the limit since the (Sub)TimeFrames are written as a whole.

**--data-sink-sidecar**
:   Write a sidecar file for each (Sub)TimeFrame file containing information about data blocks
    written in the data file. Note: Useful for debugging.
    *Warning: Format of sidecar files is not stable. This option is for debugging only.*

# NOTES

To enable zero-copy operation using shared memory, make sure the parameter **--transport** is set
to '*shmem*' and that all input and output channels are of '*shmem*' type as well. Also, consider
setting the **--io-threads** parameter to a value equal to, or lower than, the number of CPU cores
on your system.


