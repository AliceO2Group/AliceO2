<!-- doxy
\page refFrameworkCoreCOOKBOOK Core COOKBOOK
/doxy -->

# Data Processing Layer Cookbook

This is a work in progress entrypoint for common DPL related tasks.

#### Utilities for working with `DataRef`

Get payload size:
```cpp
size_t payloadSize = DataRefUtils::getPayloadSize(ref);
```

Extract a header from the header stack:
```cpp
const HeaderType* header = DataRefUtils::getHeader<HeaderType*>(ref);
```

Get the payload of messageable type as `gsl::span`
```cpp
auto data = DataRefUtils::as<T>(ref);
for (const auto& element : data) {
  // do something on element, remember it's const
}
```

#### How do I report failure for a given Algorithm?

Whenever the driver process spots an error message, i.e. an error printed via `LOG(ERROR)` facility, when the driver process quits, it will exit with a exit code of 1. This includes any exception reported by the default exception handler.

This comes handy, for example in tests.

### Debugging on your laptop

The way the DPL currently works is that the driver executable you launch,
will then take care of spawning one device per `DataProcessorSpec` in
a separate process. This means that in order to debug your code you need to
make sure gdb / lldb are actually debugging the right child process.

For `gdb` you can use the `follow-fork-mode` setting. See
[here](https://sourceware.org/gdb/onlinedocs/gdb/Forks.html) for the full
documentation. This is unfortunately not available in
[lldb](https://bugs.llvm.org/show_bug.cgi?id=17972).

Alternatively you can start your driver executable with the `-s` / `--stop`
command line option which will immediately stop execution of the children after
the fork, allowing you to attach to them, e.g. for gdb using:

```bash
attach <pid>
```

or the `lldb` equivalent:

```bash
attach -pid <pid>
```
Note: On some systems, attaching might fail due to missing permission, and `gdb`
has to be started with `sudo`.

In case you are building the DPL with the support for the debug GUI, you can
also attach a debugger to the running process by clicking on the
DataProcessorSpec you want to debug, which will show the Device inspector on
the right, and there click on the "Attach debugger" button. By default this
will start lldb in a Terminal.app window on mac, or GDB in an xterm elsewhere.
You can customise this behavior by setting the environment variable
`O2DPLDEBUG` to the command you want to run for the debugger and use the
environment variable `O2DEBUGGEDPID` to get the PID of the DataProcessor
currently selected. You can do this multiple times for all the devices you
wish to debug, but remember that you will need to quit the debugger if you want
DPL to exit.

On linux you might need to start the debugger with `sudo` to have the permission
to attach, e.g. set O2DPLDEBUG to
```bash
export O2DPLDEBUG='xterm -hold -e sudo gdb attach $O2DEBUGGEDPID &'
```
Be sure to use single quotes to avoid direct expansion of O2DEBUGGEDPID variable.
The `&` character add the end is needed to start gdb in a separate process.

### Dumping stacktraces on a signal

If you are on linux you can get stacktraces on a various signals via the:

```
--stacktrace-on-signal "<signal> [<signal>..]"
```

option, where `<signal>` can be: all, segv, bus, ill, abrt, fpe and sys.


### Debug GUI

The demonstator also includes a simple GUI to help debugging problems:

![](https://user-images.githubusercontent.com/10544/29307499-75bb8550-81a2-11e7-9aa6-96b7613288b5.png)

The GUI provides the following facilities:

* Graph view with all the connections between DataProcessors
* One log window  per DataProcessor, allowing filtering and  triggering on log
  messages
* Metrics inspector

### Integrating with non-DPL devices

Given the Data Processing Layer comes somewhat later in the design of O2, it's possible that you already have some topology of devices which you want to integrate, without having to port them to the DPL itself. Alternatively, your devices might not satisfy the requirements of the Data Processing Layer and therefore require a "raw" `FairMQDevice`, fully customised to your needs. This is fully supported and we provide means to ingest foreign, non-DPL FairMQDevices produced, messages into a DPL workflow. This is done via the help of a "proxy" data processor which connects to the foreign device, receives its inputs, optionally converts them to a format understood by the Data Processing Layer, and then pumps them to the right Data Processor Specs. In order to have such a device in your workflow, you can use the [`specifyExternalFairMQDeviceProxy`][specifyExternalFairMQDeviceProxy] helper to instanciate it. For an example of how to use it you can look at
[`Framework/TestWorkflows/src/test_RawDeviceInjector.cxx`][rawDeviceInjectorExample]. The `specifyExternalFairMQDeviceProxy` takes four arguments:

```cpp
specifyExternalFairMQDeviceProxy("foreign-source",
                {outspec},
                "type=sub,method=connect,address=tcp://localhost:5450,rateLogging=1",
                o2DataModelAdaptor(outspec, 0, 1)
               ),
```

the first one is the usual `DataProcessorSpec` name, the second one is a list of outputs which we will create from the non-DPL device, the third one is a string to connect to the existing topology and a the fourth one is a function of the kind `o2::framework::InjectorFunction` which does the actual conversion. In this particular case we use the `o2DataModelAdaptor()` helper to create such an translation function since we know that our input is already respecting the O2 Data Model and most of the heavylifing can be done automatically.

Sending out the results of a computation can be done in a similar manner. Use

```cpp
ConfigParamSpec{"channel-config", VariantType::String, "<channel-configuration>", "Out-of-band channel config"}
```

to create an out-of-band channel as specified in `channel-configuration` and hen use the `RawDeviceService` to get the raw FairMQDevice and send data hrough such a channel.

## Customizing workflows creation (WIP)

Sometimes it's handy to customise or generalise the workflow creation based on
external inputs. For example you might want to change the number of workers for
a given task or disable part of the topology if a given detector should not be
enabled. 

This can be done by implementing the function:

```cpp
void customize(std::vector<o2::framework::ConfigParamSpec> &workflowOptions)
```

**before** including the `Framework/runDataProcessing.h` (this will most likely 
change in the future). Each ConfigParamSpec will be added to the configuration
mechanism (e.g. the command line options) allowing you to modify them. Such options
will then be made available at workflow creation time via the `ConfigContext`
passed to the `defineDataProcessing` function, using the `ConfigContext::options()`
getter.

## Completion policies (WIP)

By default the data processing of a given record happens when all the fields of
that record are present. So if your Data Processor declares it will consume
`TRACKS` and `CLUSTERS`, for any given time interval both need to be produced
by some other data processor before the computation declared in yours can happen.

Sometimes it's however desirable to customise such a behavior, so that some action
on the record can happen even if it's not complete. For example you might want
to start computing some quantity as a given message arrives and then complete the
computation once the record is complete. This is done by specifying by customising 
the data processing CompletionPolicy. This can be done using the usual **Customization
mechanism** where a:

```cpp
void customize(std::vector<CompletionPolicy> &policies);
```

function is provided before including `runDataProcessing.h`.

Each `CompletionPolicy` requires the user to specify the `matcher` to select
which device is affected by it, and a `callback` to decide what action
expressed by a `CompletionOp` to take on a given input record.

Possible actions include:

* `CompletionPolicy::CompletionOp::Consume`: run the data processing callback and 
  mark the available fields in the input as consumed.
* `CompletionPolicy::CompletionOp::Process`: run the data processing callback, but do
  not consume the field, which will be available when the next message for the field
* `CompletionPolicy::CompletionOp::Wait`: hold on the record but do not process it yet.
* `CompletionPolicy::CompletionOp::Drop`: drop the current available fields from the record.

The default completion policy is to consume all inputs when they are all present.

When the computation is dispatched with a partially completea `InputRecord`,
the user can check for the validity of any of its parts via the `InputRecord::isValid()`
API.

## Customizing deployment configuration (WIP)

By default every device instanciated by the Data Processing Layer connects to
the others using the PUB/SUB paradigm. This might or might not be desiderable
for some or even all of the connections. For this reason there is now a way to
customise the connections based on the ids of the devices being instanciated.

In order to do so, one needs to implement the function

```cpp
customize(std::vector<o2::framework::ChannelConfigurationPolicy> &policies)
```

**before** including `Framework/runDataProcessing.h` (this will most likely
change in the future). You can then extend the policies vector with your own 
`ChannelConfigurationPolicy`. For each device to device edge, the system will
invoke the `ChannelConfigurationPolicy::match` callback with the ids of the
producer and of the consumer as arguments. If the callback returns `true`,
the `ChannelConfigurationPolicy::modifyInput` and
`ChannelConfigurationPolicy::modifyOutput` will be invoked passing the input and 
output channel associated to the two devices, giving the opportunity to modify 
the matching channels.

## Getting objects from the CCDB

In order to get objects from the CCDB one can specify the `Lifetime::Condition`
for the required input spec. That will retrieve the object not from another
data processor but it will do a request to a CCDB server. The actual URL for
the server can be specified via the `--condition-backend <backend-url>` option.
It is also possible to specify a given timestamp for the object via the option
`--condition-timestamp <timestamp>`. The final url is completed by the value of
the the Origin and Description of the `InputSpec` to be:

```bash
<backend-url>/<origin>/<description>/<timestamp>
```

If the timestamp is not specified, DPL will look it up in the `DataProcessingHeader`.

# Future features

## Lifetime support

While initially foreseen in the design, Lifetime for Inputs / Outputs has not
yet being implemented correctly. However, once that happens, the following behaviors 
will be implemented (naming foreseen to change). Once implemented it will be possible
to specify the following Lifetime types:

* Timeframe: an input that gets processed only once.
* Condition: an input that is cache on considered valid for multiple computations,
             according to its IOV.
* Transient: an output which is not actually sent. Useful to use the same mechanism
             of the Message Passing API to create
* QA: an output which once send is also proposed as input to the subsequent computation,
      allowing for accumulating data (e.g. histograms).
* SubTimeframe: an input which gets processed only once which has a
                granularity of less than a timeframe. Within one computation
                multiple of these can be created. They get sent as soon as
                they go out of scope.

## Wildcard support for InputSpec / OutputSpec

In order to reduce the amount of code which one has to write to define inputs
and outputs, we plan to make the InputSpecs and OutputSpecs as veritable
matchers, supporting wildcards. For example if your Algorithm supports
processing clusters coming from multiple detectors, it will be possible to
specify:

```cpp
InputSpec{"*", "CLUSTERS"}
```

If the user wants to get both clusters and tracks coming from the same detector,
it will be possible to write:

```cpp
InputSpec{"*", "CLUSTERS"}, InputSpec{"*", "TRACKS"}
```

i.e. the first message which arrives will define the wildcard for all the other input
spec in the definition.

### Data flow parallelism

Data flow parallelism is simply expressed by tuning the data flow, adding explicitly the parallel data paths, using the appropriate `InputSpec` and `OutputSpec`.

E.g.:

```cpp
DataProcessorSpec{
  "tpc_processor_1",
  Inputs{},
  Outputs{{"TPC", "CLUSTERS", SubSpec(0)}},
  // ...
},
DataProcessorSpec{
  "tpc_processor_2",
  Inputs{},
  Outputs{{"TPC", "CLUSTERS", SubSpec(1)}},
  // ...
}
// ...
DataProcessorSpec{
  "tpc_processor_18",
  Inputs{},
  Outputs{{"TPC", "CLUSTERS", SubSpec(17)}},
  // ...
}
```

or alternatively the parallel workflows part could be generated programmatically:

```cpp
parallel(
  DataProcessorSpec{
    "tpc_processor",
    {InputSpec{"c", "TPC", "CLUSTERS"}}
  },
  18,
  [](DataProcessorSpec &spec, size_t idx) {
    spec.outputs[0].subSpec = idx; // Each of the 18 DataProcessorSpecs should have a different subSpec
  }
)
```

Similarly this can be done for a component that merges inputs from multiple parallel devices, this time by modifying programmatically the `Inputs`:

```cpp
// ...
DataProcessorSpec{
  "merger",
  mergeInputs({"a", "TST", "A", 0, Lifetime::Timeframe},
              4,
              [](InputSpec &input, size_t index) {
                 input.subSpec = index;
              }
          ),
  {},
  AlgorithmSpec{[](InitContext &setup) {
     return [](ProcessingContext &ctx) {
  // Create a single output.
    LOG(DEBUG) << "Invoked" << std::endl;
  };
}
// ...
```

When one declares a parallel set of devices you can retrieve the rank (i.e. parallel id) or the number of parallel devices by using the `ParalleContext`, which can be retrieved from the `ServiceRegistry` (see also the `Services` section below), e.g.:

```cpp
size_t whoAmI = services.get<ParallelContext>().index1D();
size_t howManyAreWe = services.get<ParallelContext>().index1DSize();
```

### Time pipelining

A second type of parallelism is time based pipelining. This assumes that the data can be subdivided in subsequent "time periods" that are independent one from the other and which are each identified by some timestamp entity. In this particular case it could result handy that some part of the workflow are actually processing different time periods. This can be expressed via the `timePipeline`, directive, e.g.:

```cpp
// ...
timePipeline(DataProcessorSpec{
  "processor",
  {InputSpec{"a", "TST", "A"}},
  {OutputSpec{"TST", "B"}},
  AlgorithmSpec{[](ProcessingContext &ctx) {
    };
  }
}, 2);
// ...
```

which will result in two devices, one for even time periods, the other one for
odd timeperiods. This can also be achieved on the command line via the `--pipeline <processor name>:<N>` option, e.g. `--pipeline processor:2` in this case.

You can get programmatically the number of time pipelined devices you belong and the rank by looking it up in the `DeviceSpec`, e.g.:

```cpp
ctx.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
ctx.services().get<const o2::framework::DeviceSpec>().maxInputTimeslices;
```

Where ctx is either the ProcessingContext or the InitContext.


### Vectorised input

Sometimes data processing requires to group together multiple messages in one single multipart vector, so that they can be multiplexed on the same InputSpec. This is in particular the case for the RAW data coming out of the (Sub)TFBuilder.
In order to do so you need to make sure that the sender sends all the parts to be multiplexed in a single go. On the receiving side, you will get a single entry in the InputRecord and you can get the number of combined parts via `InputRecord::getNoParts()`. You can each of the parts by providing the entra parameter parts to the `InputRecord::get()` method.

### Using command line options in DataProcessorSpec

Command line options for a given DataProcessorSpec are defined as a std::vector\<ConfigParamSpec\>.

A ConfigParamSpec is defined by the 4 arguments
  * name
  * type
  * default
  * help

or with a constructor using only 3 arguments (without the default value).

E.g.
```cpp
  { {"opt1", VariantType::String, "def1", {"Command line option 1"}},    // constructor with default value def1
    {"opt2", VariantType::Int, {"Command line option 2"}},               // constructor without default value  
    {"opt3", VariantType::Float, 10., {"Command line option 3"}} }
```
    
(the available VariantType are listed in Framework/Variant.h).

The options are internally filled into an object of type ConfigParamRegistry and forwarded to the InitCallback of the respective AlgorithmSpec as part of the argument of type InitContext. The ConfigParamRegistry is finally accessed with InitContext::options().

E.g.
```cpp
ConfigParamRegistry opts = ic.options();  // with InitContext ic;
```


ConfigParamRegistry has the two methods `isSet(key)` and `get<T>(key)`.  

To read the option value use the `get<T>` method with the appropriate type `T`, e.g.

```cpp
  auto vopt1 = ic.options().get<std::string>("opt1");
  auto vopt2 = ic.options().get<std::int>("opt2");
```

To test wether the option `key` was set on the command line the method `isSet(key)` can be used. However be aware that the method `isSet(key)` with an option defined with a default value (constructor with
4 arguments) will always return `true`. If the option was set on the command line, then it will have the respective set value. If it is not set on the command line then it will have the default value. On
the other hand an option defined without a default value (constructor with 3 arguments) will only be recognized as set if it indeed was set on the command line. If it was not set, then its value will be
undefined. Thus to read an option without default value do e.g.

```cpp
  std::string vopt1("");
  if (ic.options().isSet("opt1")) {
    vopt1 = ic.options().get<std::string>("opt1");
  }
```

## Monitoring

By default DPL exposes the following metrics to the back-end specified with:
`--monitoring-backend`:

* `malformed_inputs`: number of messages which did not match the O2 DataModel
* `dropped_computations`: number of messages which DPL could not process
* `dropped_incoming_messages`: number of messages which DPL could 
                             not accept in its own queue.
* `relayed_messages`: number of messages received by DPL.

* `errors`: number of errors recorded inside DPL (not in the actual processing).
* `exceptions`: number of exceptions raised by the DPL.
* `inputs/relayed/pending`: number of entries in the DPL queue which are waiting for extra data.
* `inputs/relayed/incomplete` : 1 if the device is waiting for extra data.
* `inputs/relayed/total`: how many inputs the processor has.
* `elapsed_time_ms`:
* `last_processed_input_size_byte`: how many bytes were processed on last iteration
* `total_processed_input_size_byte`: how many bytes were processed in total since the beginning
* `last_processing_rate_mb_s`: at what rate the last message was processed
* `min_input_latency_ms`: the shortest it took for any message to be processed by this dataprocessor (since created)
* `max_input_latency_ms`: the maximum it took for any message to be processed by this dataprocessor (since created)
* `input_rate_mb_s`: 

Moreover if you specify `--resources-monitoring <poll-interval>` the 
process monitoring metrics described at:

<https://github.com/AliceO2Group/Monitoring/#process-monitoring>

will be pushed every `<poll-interval>` seconds to the same backend and dumped in the `performanceMetrics.json` file on exit.

### Disabling monitoring

Sometimes (e.g. when running a child inside valgrind) it might be useful to disable metrics which might pollute STDOUT. In order to disable monitoring you can use the `no-op://` backend:

```bash
some-workflow --monitoring-backend=no-op://
```

notice that the GUI will not function properly if you do so.
