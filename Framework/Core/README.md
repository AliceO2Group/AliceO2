# Data Processing Layer in O2 Framework

## Status quo and motivation for an O2 Data Processing Layer

FairMQ  currently provides  a well  documented and  flexible framework  for an
actor  based computation  where each  of the  actors listens  for message-like
entities on channels  and executes some code as a  reaction. The key component
which controls this is called a `FairMQDevice` (or *device* from now on) which
can use different kind of transports to receive and send messages. In the most
generic case,  users are  allowed to  have full control  on the  state machine
governing the  message passing and  have complete  control on how  the message
handling  is done.  This of  course  covers all  ALICE usecases  in a  generic
manner, at the cost of extra complexity left to the user to implement. In most
cases however a  simplified way of creating devices is  provided, and the user
will simply create  its own `FairMQDevice`-derived class,  which registers via
the `OnData(FairMQParts &parts)` method a  callback that is invoked whenever a
new message arrives. This however still holds the user responsible for:

- Verifying that  the required inputs  for the computation are  all available,
  both  from  the actual  data  flow  (being  it for  readout,  reconstruction
  or  analysis)  and   from  the  asynchronous  stream   (e.g.  alignment  and
  calibrations).
- Create the appropriate message which holds the results and send it.
- Ensure the ease of testability, the  code reuse and the proper documentation
  `OnData` callback.  In particular there is  no way to inspect  which data is
  expected by a device and which data is produced.

This is by design, because the FairMQ transport layer should not know anything
about the  actual data being transferred,  while all the points  above require
some sort  of inner knowledge  about the data model  and the data  being moved
around.

The aim is to achieve the following:

- **Explicit data  flow:** Input and outputs  are declared upfront and  can be
  used  for documentation  or for  automatic topology  creation (assuming  the
  actual processing environment is known).
- **Transport agnostic  data processing:** users  will not have to  know about
  the details of  how the data materialises on their  device, they will always
  be handed  the full  set of payloads  they requested, even  if they  come at
  different time.
- **Composability  of data  processing:** different  process functions  can in
  principle  be  chained  and  scheduled together  depending  on  the  desired
  granularity for devices.

## Separating data-processing from transport

For the reasons  mentioned above, we propose that the  one of the developments
which should happen with the O2 Framework work package is the development of a
“Data Processing layer” which actually knows  about the O2 Data Model (and for
this reason  cannot be  part of  FairMQ itself) and  exploits it  to validate,
optimise  and correctly  schedule a  computation on  a user  specified set  of
inputs.

The Data Processing Layer in particular requires:

- That the inputs of each computation are provided upfront.
- That the outputs of each computation are provided upfront.
- That a time identifier can be associated to inputs

and given these premises it actually guarantees:

- That no computation is performed before all the inputs for a given time
  identifier are available
- That no message passing happens during the performing of the computation,
  but uniquely at the end.

### Instanciating a computation

The description of  the computation in such the Data  Processing Layer is done
via instances  of the [`DataProcessorSpec`][DataProcessorSpec]  class, grouped
in a  so called `WorkflowSpec` instance.  In order to provide  a description a
computation to  be run, the user  must implement a callback  which modifies an
empty `WorkflowSpec` instance provided by the system. E.g.:

```cpp
#include "Framework/Utils/runDataProcessing.h"

void defineDataProcessing(WorkflowSpec &workflow) {
  auto spec = DataProcessorSpec{
    ...
  };
  // Fill a DataProcessingSpec "spec"
  workflow.push_back(spec);
}
```

See   next    section,   for    a   more    detailed   description    of   the
[`DataProcessorSpec`][DataProcessorSpec]  class.  The  code above  has  to  be
linked into a  single executable together with the Data  Processing Layer code
to form a so called driver executable which if run will:

- Map all [`DataProcessorSpec`][DataProcessorSpec] to a set of `FairMQDevice`s
  (using 1-1 correspondence, in the current implementation).
- Instanciate and start all the devices resulted from the previous step.
- (Optionally) start a GUI which allows to monitor the running of the system.

[DataProcessorSpec]: https://github.com/AliceO2Group/AliceO2/blob/dev/Framework/Core/include/Framework/DataProcessorSpec.h

### Describing a computation

The  description  of   the  computation  in  such  a  layer   is  done  via  a
`DataProcessorSpec` class, which  describes some sort of processing  of a (set
of) O2  Data Payloads  (*payloads* from  now on),  as defined  by the  O2 Data
Model,  eventually  producing new  payloads  as  outputs.  The inputs  to  the
computation,  the  outputs  and the  actual  code  to  run  on the  former  to
produce the latter,  is specified in a  `DataProcessorSpec` instance. Multiple
`DataProcessorSpec` instances can be grouped  together in a `WorkflowSpec`. to
the  driver code  which  maps configures  the  processing device  accordingly.
Practically  speaking  this would  translate  into  something similar  to  the
current  “simplified device  mode” where  the user  includes a  special header
which contains all the boilerplate and  provides a well defined callback where
the requested `DataProcessorSpec` are provided.

The `DataProcessorSpec` is defined as follows:

```cpp
struct DataProcessorSpec {
   using InitCallback = std::function<ProcessCallback(InitContext &)>;
   using ProcessCallback = std::function<void(ProcessingContext &)>;
   using ErrorCallback = std::function<void(ErrorContext &)>;
   std::vector<InputSpec> inputs;
   std::vector<OutputSpec> outputs;
   std::vector<ConfigParamSpec> configParams;
   std::vector<std::string> requiredServices;
   AlgorithmSpec algorithm;
};
```

In the above both `InputSpec` and `OutputSpec` are like:

```cpp
struct InputSpec {              // OutputSpec as well
  o2::Headers::DataDescription description;
  o2::Headers::DataOrigin origin;
  o2::Headers::SubSpecificationType subSpec;
  enum Lifetime lifetime;
};
```

where  description, origin  and subSpec  match the  O2 Data  Model definition.
For  the  moment  we  will  consider  this a  one  to  one  mapping  with  the
`o2::Headers::DataHeader` ones. In principle one  could think of a one-to-many
relationship (e.g. give  me all the clusters, regardless  of their provenance)
and the processing layer could automatically aggregate those in a unique view.
This is also the semantic difference between `InputSpec` and `OutputSpec`: the
former is to express  data that matches a given query (which  must be exact at
the moment) the latter is to describe in all details and without any doubt the
kind of the produced outputs.

The `lifetime` property:

```cpp
enum Lifetime {
  Timeframe,
  Condition,
  QA,
  Transient
};
```

can be  used to  distinguish if  the associated  payload should  be considered
payload data, and therefore be processed  only once, or alignment / conditions
data, and  therefore it  would be considered  valid until a  new copy  is made
available to the device.

The `configParams` vector would be used to specify which configuration options
the data processing being described requires:

```cpp
struct ConfigParamSpec {
  std::string name;
  enum ParamType type;
  variant defaultValue;
};
```

command line /  configuration options would be automatically  generated by it.
These are available only at init stage, and can be used to configure services.
They are  not available to the  actual `process` callback as  all the critical
parameters  for data  processing should  be part  of the  data stream  itself,
eventually coming from CCDB / ParameterManager.

Similarly  the  `requiredServices`  vector  would define  which  services  are
required for  the data processing. For  example this could be  used to declare
the need for some data cache, a GPU context, a thread pool.

The  `algorithm`  property, of  `AlgorithmSpec`  is  instead used  to  specify
the  actual computation.  Notice that  the same  `DataProcessorSpec` can  used
different `AlgorithmSpec`.  The rationale  for this is  that while  inputs and
outputs might  be the same,  you might want  to compare different  versions of
your algorithm. The `AlgorithmSpec` resembles the following:

```cpp
struct AlgorithmSpec {
  using ProcessCallback = std::function<void(ProcessingContext &)>;
  using InitCallback = std::function<ProcessCallback(InitContext &)>;
  using ErrorCallback = std::function<void(ErrorContext &)>;

  InitCallback onInit = nullptr;
  ProcessCallback onProcess = nullptr;
  ErrorCallback onError = nullptr;
};
```

The `onProcess` function is to be used for stateless computations. It’s a free
function and  it’s up  to the  framework to  make sure  that all  the required
components are declared upfront. It takes as input the context for the current
computation in  the form of a  `ProcessingContext &` instance. Such  a context
consist of:

- An `InputRecord` which allows retrieving the current inputs matching the provided 
  specification.
- A `ServiceRegistry` referencing the set of services it declared as required
  the computation.
- A `DataAllocator` allocator which can allocate new payloads only for the
  types which have been declared as `outputs`.

`onProcess` is  useful whenever  your computation is  fully contained  in your
input. In several cases, however, a computation requires some ancillary state,
which needs to be  initialised only on (re-)start of the  job. For example you
might want to initialise the geometry of  your detector. To do so, you can use
the  `onInit` callback  and allocate  the state  and pass  it to  the returned
`ProcessCallback` as captured arguments. E.g:

```cpp
AlgorithmSpec{
  InitCallBack{[](InitContext &setup){
      auto statefulGeo = std::make_shared<TGeo>();
      return [geo = statefulGeo](ProcessingContext &) {
        // do something with geo
      };
    }
  }
}
```

A `DataRef` would look like:

```cpp
struct DataRef {
  const InputSpec *spec;
  const char *const header;
  const char *const payload;
};
```    

`header` and `payload`  are the pointers to the data  which matches the `spec`
InputSpec.

## Implementing a computation

This chapter describes how to actually implement an `AlgorithmSpec`.

### Using inputs - the `InputRecord` API

Inputs   to   your   computation   will   be   provided   to   you   via   the
[`InputRecord`][InputRecord] API. An instance of  such a class is hanging from
the `ProcessingContext`  your computation  lambda is  passed and  contains one
value for each of the `InputSpec` you specified. E.g.:

```cpp
InputRecord &args = ctx.inputs();
```

From the `InputRecord` instance you can get the arguments either via their positional
index:

```cpp
DataRef ref = args.getByPos(0);
```

or using the mnemonics-label which was used as first argument in the associated
`InputSpec`.

```cpp
DataRef ref = args.get("points");
```

You can then use the `DataRef` `header` and `payload` raw pointers to access
the data in the messages.

If the message is of a known type, you can automatically get a casted reference
to the contents of the message by passing it as template argument, e.g.:

```cpp
XYZ &p = args.get<XYZ>("points");
```

[InputRecord]: https://github.com/AliceO2Group/AliceO2/blob/HEAD/Framework/Core/include/Framework/InputRecord.h

### Creating outputs - the DataAllocator API

In order  to prevent  algorithms  to  create  data they  are  not  supposed to
create, a special `DataAllocator` object is passed to the process callback, so
that only messages for declared outputs  can be created. A `DataAllocator` can
create Framework owned resources via the `make<T>` method. In case you ask the
framework to create a collection of  objects, the result will be a `gsl::span`
wrapper around the collection. A  `DataAllocator` can adopt externally created
resources via  the `adopt` method. A  `DataAllocator` can create a  copy of an
externally owned resource via the `snapshot` method.

Currently supported data types are:

- Vanilla `char *` buffers with associated size. This is the actual contents of
  the FairMQ message.
- POD types. These get directly mapped on the message exchanged by FairMQ and 
  are therefore "zerocopy" for what the DataProcessingLayer is concerned.
- POD collections, which are exposed to the user as `gsl::span`.
- TObject derived classes. These are actually serialised via a TMessage
  and therefore are only suitable for the cases in which the cost of such a
  serialization is not an issue.

Currently supported data types for snapshot functionality, the state at time of
calling snapshot is captured in a copy:
- POD types
- TObject derived classes, serialized
- std::vector of POD type, at the receiver side the collection is exposed
  as gsl::span
- std::vector pointer to POD type, the objects are linearized in the message
  and exposed as gsl::span on the receiver side

The DataChunk object resembles a `iovec`:

```cpp
struct DataChunk {
  char *data;
  size_t size;
};
```

however, no API is provided to explicitly send it. All the created DataChunks
are  sent (potentially  using scatter  / gather)  when the  `process` function
returns. This  is to avoid  the “modified after  send” issues where  a message
which was sent is still owned and modifiable by the creator.

### Error handling

When  an error  happens during  processing  of some  data, the  writer of  the
`process` function should simply throw  an exception. By default the exception
is  caught  by  the  `DataProcessorManager`  and  a  message  is  printed  (if
`std::exeception` derived `what()` method is used, otherwise a generic message
is given). Users can provide themselves an error handler by specifying via the
`onError` callback specified in `DataProcessorSpec`.

### Services

Services  are utility  classes which  `DataProcessor`s can  access to  request
out-of-bound,  deployment dependent,  functionalities. For  example a  service
could  be used  to post  metrics to  the  monitoring system  or to  get a  GPU
context. The  former would  be dependent  on whether you  are running  on your
laptop (where  monitoring could simply mean  print out metrics on  the command
line) or in  a large cluster (where monitoring probably  means to send metrics
to an aggregator device which then pushes them to the backend.

Services  are initialised  by  the driver  code (i.e.  the  code included  via
runDataProcessing.h) and passed to the  user code via a `ServiceRegistry`. You
can  retrieve  the service  by  the  type of  its  interface  class. E.g.  for
monitoring you can do:

```cpp
#include "Framework/MetricsService.h"
// ...
auto service = ctx.services().get<MetricsService>(); // In the DataProcessor lambda...
service.post("my/metric", 1); ...
```

Currently available services are described below.

#### ControlService

The control service allow DataProcessors to modify their state or the one of
their peers in the topology. For example if you want to quit the whole data
processing topology, you can use:

```cpp
#include "Framework/ControlService.h"
//...
auto ctx.services().get<ControlService>().readyToQuit(true) // In the DataProcessor lambda
```

#### RawDeviceService

This service allows you to get an hold of the `FairMQDevice` running the
DataProcessor computation from with the computation itself. While in general
this should not be used, it is handy in case you want to integrate with a
pre-existing `FairMQDevice` which potentially does not even follow the O2 Data
Model.

## Miscellaneous topics

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

### Expressing parallelism

If we want to retain a message passing semantic and really treat shared memory
as  yet another  transport, we  need  to be  very  careful in  how to  express
parallelism on data,  so that the “single ownership model”  of message passing
forces us to either duplicate streams that need to be accessed in parallel, or
to  serialise workers  which  need to  access the  same  data. Solutions  like
reference counting shared  memory would not be allowed in  such a scenario and
in any case would require extra caution and support to make sure that failures
do not leave dangling reference around  (e.g. when one of the parallel workers
abruptly terminates). First of all let’s  consider the fact that there are two
level of parallelisms which can be achieved:

- Data flow  parallelism: when data  can be  split in partitions  according to
  some subdivision criteria (e.g. have one  stream per TPC sector and have one
  worker for each).
- Time flow parallelism: when parallelism can be achieved by having different
  workers handle different time intervals for the incoming data. (e.g. worker 0
  processes even timeframes, worker 1 processes odd timeframes).

Data flow parallelism is simply expressed by tuning the data flow, adding
explicitly the parallel data paths, using the appropriate `InputSpec` and
`OutputSpec`. E.g.:

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

or alternatively the parallel workflows part could be generated
programmatically:

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

Similarly this can be done for a component that merges inputs from multiple
parallel devices, this time by modifying programmatically the `Inputs`:

```cpp
// ...
DataProcessorSpec{
  "merger",
  mergeInputs({"a", "TST", "A", InputSpec::Timeframe},
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

When one declares a parallel set of devices you can retrieve the rank (i.e.
parallel id) or the number of parallel devices by using the `ParalleContext`,
which can be retrieved from the `ServiceRegistry` (see also the `Services`
section below), e.g.:

    size_t whoAmI = services.get<ParallelContext>().index1D();
    size_t howManyAreWe = services.get<ParallelContext>().index1DSize();

A second type of parallelism is time based pipelining. This assumes that the
data can be subdivided in subsequent "time periods" that are independent one
from the other and which are each identified by some timestamp entity. In this
particular case it could result handy that some part of the workflow are
actually processing different time periods. This can be expressed via the
`timePipeline`, directive, e.g.:

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
odd timeperiods.


### Debug GUI

The demonstator also includes a simple GUI to help debugging problems:

![](https://user-images.githubusercontent.com/10544/29307499-75bb8550-81a2-11e7-9aa6-96b7613288b5.png)

The GUI provides the following facilities:

- Graph view with all the connections between DataProcessors
- One log window  per DataProcessor, allowing filtering and  triggering on log
  messages
- Metrics inspector

### Integrating with pre-existing devices

Given the Data Processing Layer comes somewhat later in the design of O2, it's
possible that  you already  have some  topology of devices  which you  want to
integrate, without having to port them  to the DPL itself. Alternatively, your
devices might  not satisfy the requirements  of the Data Processing  Layer and
therefore  require a  "raw" `FairMQDevice`,  fully customised  to your  needs.
This  is fully  supported  and we  provide means  to  ingest foreign,  non-DPL
FairMQDevices produced,  messages into a  DPL workflow.  This is done  via the
help of  a "proxy" data processor which  connects to the foreign  device, receives its
inputs, optionally converts them to a format understood by the Data Processing
Layer, and then pumps them to the right Data Processor Specs. In order to have
such a device in your workflow, you can use the
[`specifyExternalFairMQDeviceProxy`][specifyExternalFairMQDeviceProxy] helper to instanciate it.
For an example of how to use it you can look at
[`Framework/TestWorkflows/src/test_RawDeviceInjector.cxx`][rawDeviceInjectorExample].
The `specifyExternalFairMQDeviceProxy` takes four arguments:

```cpp
specifyExternalFairMQDeviceProxy("foreign-source",
                {outspec},
                "type=sub,method=connect,address=tcp://localhost:5450,rateLogging=1",
                o2DataModelAdaptor(outspec, 0, 1)
               ),
```

the first one is the usual `DataProcessorSpec` name, the second one is a list
of outputs which we will create from the non-DPL device, the third one is a
string to connect to the existing topology and a the fourth one is a function
of the kind `o2::framework::InjectorFunction` which does the actual conversion.
In this particular case we use the `o2DataModelAdaptor()` helper to create such an
translation function since we know that our input is already respecting the O2
Data Model and most of the heavylifing can be done automatically.

Sending out the results of a computation can be done in a similar manner. Use

```cpp
ConfigParamSpec{"channel-config", VariantType::String, "<channel-configuration>", "Out-of-band channel config"}
```

to create an out-of-band channel as specified in `channel-configuration` and
then use the `RawDeviceService` to get the raw FairMQDevice and send data
through such a channel.

[specifyExternalFairMQDeviceProxy]: https://github.com/AliceO2Group/AliceO2/blob/dev/Framework/Core/include/Framework/ExternalFairMQDeviceProxy.h
[rawDeviceInjectorExample]: https://github.com/AliceO2Group/AliceO2/blob/dev/Framework/TestWorkflows/src/test_RawDeviceInjector.cxx

## Current Demonstrator (WIP)

An demonstrator illustrating a possible implementation of the design described
above is now found in the dev branch of AliceO2, in the
[Framework](https://github.com/AliceO2Group/AliceO2/tree/dev/Framework) folder.
In particular:

- `Framework/Core` folder contains the `DataProcessorSpec` class and related.
- `Framework/Core/test` folder contains a few unit test and simple example workflows.
- `Framework/TestWorkflows` folder contains a few example workflows.
- `Framework/DebugGUI` folder contains the core GUI functionalities.


## Interesting reads

- [MillWheel:     Fault-Tolerant     Stream     Processing     at     Internet
  Scale](https://research.google.com/pubs/pub41378.html) :  paper about Google
  previous generation system for stream processing
- [Concord](http://concord.io)  : Similar  (to  the  above) stream  processing
  solution, OpenSource.

## General remarks & Feedback so far:
- Gvozden and  Mikolaj were  suggesting to have  a multiple  payload view-like
  object.  Where does  that fit?  Shouldn’t this  feature be  provided by  the
  DataRef binder?
- Do we need `process` to return a `bool` / an `error` code?
- What are the possible Services we can think about?
  - CPUTimer
  - File Reader
  - File Writer
  - Monitoring
  - Logging
  - ParallelContext
- Thorsten: should configuration come always from the ccdb?
- Thorsten: should allow to specify that certain things are processed with the
            same CCDB conditions.
- Should we have different specifications for conditions objects and for data?
  - We could simplify current design by making Timeframe the default.
  - It  complicates  a bit  the  `process`  method.  We  would need  an  extra
    allocator for the output objects and we would need an extra vector for the
    inputs.
  - On the other hand  it would probably make some of the  flow code easier to
    read / more type safe.
- Can inputs  / outputs be  optional? Most likely,  no, since that  would mean
  that if  they arrive  late the  processing happens with  a different  set of
  inputs (and  consequently a  optional output means  someone has  an optional
  input). Do we need some “guaranteed delivery” for messages?
- ~~Do we need to guarantee that timeframes are processed in their natural
  order?~~ nope. Actually in general we cannot guarantee that.
- Do we want to separate “Algorithms”s from “DataProcessor”s? The former would
  declare generic argument bindings (e.g. I take x, and y) and the latter
  would do the actual binding to real data (input clusters is y, input tracks
  is y). This is what tensor flow actually does.
- Shouldn’t the DataHeader contain the timeframe ID?
- David / Ruben: sometimes the input data depends on whether
  or not a detector is active during data taking. We would therefore need a
  mechanism to mask out inputs (and maybe modules) from the dataflow, based on
  run control. If only part of the data is available, it might make sense that
  we offer “fallback” callbacks which can work only on part of the data.
- Ruben: most likely people will want to also query the CCDB directly. Does
         it make sense to offer CCDB querying as a service so that we can
         intercept (and eventually optimise) multiple queries from the same
         workflow?
- Are options scoped? I.e.  do we want to have that  if two devices, `deviceA`
  and `deviceB` defining the same option (e.g. `mcEngine`) they will require /
  support using `--``deviceA-mcEngine` and `--``deviceB-mcEngine`?
- ~~Mikolaj pointed out  that capture by move is possible  in `C++14` lambdas,
  so we should use that for the  stateful init~~. Actually this is not working
  out as expected?
