<!-- doxy
\page refDetectorsMUONMCHRawDecoder Decoder
/doxy -->

# MCH Raw Data Decoder

There are currently two data formats to be dealt with : the "bare" one coming
out of the CRU when no user logic (UL) is used, or the "UL" one where user
logic is actually used. Each of the two formats can be used in charge sum (the
sampa electronics does the charge integration) or in sample mode (the sampa
electronics transmit all the charge samples). All formats are composed of
{RawDataHeader,Payload} pairs.

Note that the cluster sum mode has to be properly selected, as there is
unfortunately no way to infer it from the data itself...

## createDecoder&lt;FORMAT,CHARGESUM,RDH>

On the reading/consumer/decoding end, the choice of the decoder to use is made
through a templatized function, `createDecoder<FORMAT,CHARGESUM,RDH>`.
Currently the following template parameters combinations have been tested : 

|      FORMAT     |   CHARGESUM   |       RDH       |
| :-------------: | :-----------: | :-------------: |
|    BareFormat   | ChargeSumMode | RawDataHeaderV4 |
|    BareFormat   |   SampleMode  | RawDataHeaderV4 |
| UserLogicFormat | ChargeSumMode | RawDataHeaderV4 |
| UserLogicFormat |   SampleMode  | RawDataHeaderV4 |

RDH V5 is not yet supported, but is planned.

As an example, to get a decoder for BareFormat in ChargeSum mode, using
RawDataHeaderV4 :

```.cpp
#include "MCHRawDecoder/Decoder.h"

RawDataHeaderHandler<RAWDataHeaderV4> rh;
SampaChannelHandler ch;

auto decoder = o2::mch::raw::createDecoder<BareFormat, ChargeSumMode, RAWDataHeaderV4>(rh, ch);

// get some memory buffer from somewhere ...
buffer = ... 

// decode that buffer
decode(buffer);
```

The `createDecoder` function requires two parameters : a `RawDataHeaderHandler`
and a `SampaChannelHandler`.
Both parameters can be defined as lambdas, regular free functions, or member
functions of some class.

## RawDataHeaderHandler

The `RawDataHeaderHandler` is a function that takes a RawDataHeader and
(optionally) returns a RawDataHeader, i.e. :

```.cpp
using RawDataHeaderHandler = std::function<std::optional<RDH>(const RDH& rdh)>;
```

If no RDH is returned (i.e. `std::nullopt` is returned) then that part of the
raw data is *not* decoded at all.
The returned RDH can be a modified version of the argument, e.g. to change some
of the values (like `feeId` according to some mapping).

## SampaChannelHandler

The `SampaChannelHandler` is also a function, that takes a dual sampa
identifier (in electronics realm, aka solar,group,index tuple), a channel
identifier within that dual sampa, a `SampaCluster` and returns nothing, i.e. :

> A word of caution here about the naming. A `SampaCluster` is a group of raw
> data samples of *one* dual sampa channel, and has nothing to do with a
> cluster of pads or something alike. 

```.cpp
using SampaChannelHandler = std::function<void(DsElecId dsId,
                                               uint8_t channel,
                                               SampaCluster)>;
```

That function is called by the raw data decoder for each `SampaCluster` that it
finds in the data.
A very simple example would be a function to dump the SampaClusters : 

```.cpp
SampaChannelHandler handlePacket(DsElecId dsId, uint8_t channel, SampaCluster sc) {
    std::cout << fmt::format("{}-ch-{}-ts-{}-q-{}", asString(dsId), channel, sc.timestamp, sc.chargeSum));
}
// (note that this particular function only correctly handles the SampaCluster in ChargeSum Mode)
```

## TODO

Add a description of the two data formats here ?
