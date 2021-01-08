<!-- doxy
\page refFrameworkUtils Utils
/doxy -->

## DPL Utilities

### Raw proxy
Executable:
```
o2-dpl-raw-proxy
```

A proxy workflow to connect to a channel from DataDistribution. The incoming
data is supposed to follow the O2 data model with header-payload pair messages.
The header message must contain `DataHeader` and the internal DPL `DataProcessingHeader`.
Only data which match the configured outputs will be forwarded.

Since payload can be split into several messages, the proxy will create a new
message to merge all parts belonging to one payload entity.

#### Usage: `o2-dpl-raw-proxy`
```
o2-dpl-raw-proxy --dataspec "0:FLP/RAWDATA" --channel-config "name=readout-proxy,type=pair,method=bind,address=ipc:///tmp/stf-builder-dpl-pipe-0,transport=shmem,rateLogging=1"
```

Options:
```
  --dataspec arg (=A:FLP/RAWDATA;B:FLP/DISTSUBTIMEFRAME/0)
                                        selection string for the data to be 
                                        proxied
  --throwOnUnmatched                    throw if unmatched input data is found
```

Data is selected by using a configuration string of the format
`tag1:origin/description/subspec;tag2:...`, the tags are arbitrary and only
for internal reasons and are ignored.
Example:
```
--dataspec 'A:FLP/RAWDATA;B:FLP/DISTSUBTIMEFRAME/0'
```

The input channel has name `readout-proxy` and is configured with option `--channel-config`.

By default, data blocks not matching the configured filter/output spec will be silently
ignored. Use `--throwOnUnmatched` to apply a strict checking in order to indicate unmatched
data.

#### Connection to workflows
The proxy workflow only defines the proxy process, workflows can be combined by piping them
together om the command line
```
o2-dpl-raw-proxy --dataspec "0:FLP/RAWDATA" --channel-config "..." | the-subscribing-workflow
```

#### TODO:
- stress test with data not following the O2 data model
- forwarding of shared memory messages
- vector handling of split payload parts in DPL
- make the channel config more convenient, do not require the full string and use
  defaults, e.g. for the name (which can not be changed anyhow)
- check if the channel config string can be parsed omitting tags, right now this
  is only possible for the first entry

### RAW parser
Class RawParser implements a parser for O2 raw data which is organized in pages of a
fixed size, each page starts with the RAWDataHeader. The page size may actually be
smaller than the maximum size, depending on the header fields.

The parser class works on a contiguous sequence of raw pages in a raw buffer.
Multiple versions of RAWDataHeader are supported transparently and selected depending
on the version field of the header.

#### Usage: `RawParser`
```
// option 1: parse method
RawParser parser(buffer, size);
auto processor = [&count](auto data, size_t length) {
  std::cout << "Processing block of length " << length << std::endl;
};
parser.parse(processor);

// option 2: iterator
RawParser parser(buffer, size);
for (auto it = parser.begin(), end = parser.end(); it != end; ++it, ++count) {
  std::cout << "Iterating block of length " << it.length() << std::endl;
  auto dataptr = it.data();
}
```

### DPL raw parser class
Utility class to transparently iterate over raw pages distributed over multiple messsages on
multiple routes.

A DPL processor might receive raw pages accumulated on three levels:
  1) the DPL processor has one or more input route(s)
  2) multiple parts per input route (split payloads or multiple input
     specs matching the same route spec
  3) variable number of raw pages in one payload

Internally, class @ref RawParser is used to access raw pages withon one
payload message and dynamically adopt to RAWDataHeader version.

The parser provides an iterator interface to loop over raw pages,
starting at the first raw page of the first payload at the first route
and going to the next route when all payloads are processed. The iterator
element is @ref RawDataHeaderInfo containing just the two least significant
bytes of the RDH where we have the version and header size.

The iterator object provides methods to access the concrete RDH, the raw
buffer, the payload, etc.

#### Usage
The parser needs an object of DPL InputRecord provided by the ProcessingContext.

```
auto& inputs = context.inputs();
DPLRawParser parser(inputs);
for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
  // retrieving RDH v4
  auto const* rdh = it.get_if<o2::header::RAWDataHeaderV4>();
  // retrieving the raw pointer of the page
  auto const* raw = it.raw();
  // retrieving payload pointer of the page
  auto const* payload = it.data();
  // size of payload
  size_t payloadSize = it.size();
  // offset of payload in the raw page
  size_t offset = it.offset();
}
```

A list of InputSpec definiions can be provided to filter the incoming messages
by the DataHeader.
```
DPLRawParser parser(inputs, o2::framework::select("A:ITS/RAWDATA"));
```
Note: `select` is a helper function of WorkflowSpec.h which builds InputSpecs from
a string.

The iterator implements stream output operator to print out RDH fields in a table,
a specific format wrapper prints RDH info and table header, e.g. for
```
std::cout << DPLRawParser::RDHInfo(it) << std::endl;
std::cout << it << std::endl;
```

output can look like
```
 DET/RAWDATA/196616   1 part(s) payload size 8192
 RDH v4
   PkC pCnt  fId  Mem CRU  EP LID    HBOrbit  HBBC  s
    92    1 4844   64   3   0   8 1013162567     0  1
```

#### Workflow example
Executable:
```
o2-dpl-raw-parser
```

Parser workflow defines one process with configurable inputs and a loop over the input
data which sets up the RawParser class and executes some basic parsing and statistic
printout.

#### Usage: `o2-dpl-raw-parser`
```
o2-dpl-raw-parser --input-spec "A:FLP/RAWDATA"
```

Options:
```
--input-spec arg (=A:FLP/RAWDATA)
--log-level n    0: off; 1: medium; 2: high  
```

In medium log level, the RDH version and the table header for the RDH field are printed
for each new data description, in level high this will be done also for all parts with the
same data description. The RDH fields are printed in a table per raw page.

The workflow can be connected to the o2-dpl-raw-proxy workflow using pipe on command line.

### DPL output proxy
Executable:
```
o2-dpl-output-proxy
```

A proxy workflow to connect to workflow output and forward it to out-of-band channels,
meaning channels outside DPL.

#### Usage: `o2-dpl-output-proxy`
```
o2-dpl-output-proxy --dataspec "channelname:TPC/TRACKS" --channel-config name=channelname,...
```

Output channel(s): to be defined with the FairMQ device channel configuration option, e.g.
```
--channel-config name=downstream,type=push,method=bind,address=icp://localhost_4200,rateLogging=60,transport=shmem
```

Input to the proxy is defined by the DPL data spec syntax
```
binding1:origin1/description1/subspecification1;binding2:origin2/descritption2;...
```
The binding label can be used internally to access the input, here it is used to match to
a configured output channel. That binding can be the same for multiple data specs.

Options:
```
  --dataspec       specs           Data specs of data to be proxied
  --channel-config config          FairMQ device channel configuration for the output channel
  --proxy-name     name            name of the proxy processor, used also as default output channel name
  --default-transport arg (=shmem) default transport: shmem, zeromq
  --default-port arg (=4200)       default port number
```
Note: the 'default-*' options are only for building the the default of the channel configuration.
The default channel name is build from the name of the proxy device. Having a default channel
configuration allows to skip the '--channel-config' option on the command line.

### ROOT Tree reader and writer
documentation to be filled
