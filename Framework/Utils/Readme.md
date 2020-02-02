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
```

The workflow can be connected to the o2-dpl-raw-proxy workflow using pipe on command line.

### ROOT Tree reader and writer
documentation to be filled
