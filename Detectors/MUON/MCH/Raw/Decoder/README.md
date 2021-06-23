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

The work of the MCH Raw Data decoder is to decode one single such pair, also
called a (CRU) page. How the loop on the raw pages is done is *not* the decoder business.

## createPageDecoder

On the reading/consumer/decoding end, the choice of the internal decoder to use
is made using the data itself. You must give (at least) a part of a raw data
buffer that contains a (valid) RawDataHeader. That RDH is used to deduce which
implementation is picked.

    gsl::span<const std::byte> rawbuffer = ... ;
    auto pageDecoder = o2::mch::raw::createPageDecoder(rawbuffer,channelHandler);

The `pageDecoder` that is returned is a function that you have to call on each
data page that you want to decode :

    while(some_data_is_available) {
    // get some memory buffer from somewhere ...
    buffer = ...

    // decode that buffer
    pageDecode(buffer);
    }

Internally the implementation is using templatized implementation, `PageDecoderImpl<FORMAT,CHARGESUM,VERSION>`.
Currently the following template parameters combinations have been tested :

|      FORMAT     |   CHARGESUM   | VERSION |
| :-------------: | :-----------: | :-----: |
|    BareFormat   | ChargeSumMode |    0    |
|    BareFormat   |   SampleMode  |    0    |
| UserLogicFormat | ChargeSumMode |    0    |
| UserLogicFormat |   SampleMode  |    0    |
| UserLogicFormat | ChargeSumMode |    1    |
| UserLogicFormat |   SampleMode  |    1    |

The `createPageDecoder` function requires two parameters : a raw memory buffer
(in the form of a `gsl::span<const std::byte>` (note that the span is on
constant bytes, i.e. the input buffer is read-only) and a
`SampaChannelHandler`.

## SampaChannelHandler

The `SampaChannelHandler` is  a function, that takes a dual sampa identifier
(in electronics realm, aka solar,group,index tuple), a channel identifier
within that dual sampa (in the 0..63 range), a `SampaCluster` and returns
nothing, i.e. :

> A word of caution here about the naming. A `SampaCluster` is a group of raw
> data samples of *one* dual sampa channel, and has nothing to do with a
> cluster of pads or something alike.

```.cpp
using SampaChannelHandler = std::function<void(DsElecId dsId,
                                               DualSampaChannelId channel,
                                               SampaCluster)>;
```

That function is called by the raw data decoder for each `SampaCluster` that it
finds in the data.
A very simple example would be a function to dump the SampaClusters :

```.cpp
SampaChannelHandler handlePacket(DsElecId dsId, DualSampaChannelId channel, SampaCluster sc) {
    std::cout << fmt::format("{}-ch-{}-ts-{}-q-{}", asString(dsId), channel, sc.timestamp, sc.chargeSum));
}
// (note that this particular function only correctly handles the SampaCluster in ChargeSum Mode)
```

## Example of decoding raw data

A (not particularly clean) example of how to decode raw data can be found in
the source of the `o2-mchraw-dump` executable [rawdump](../Tools/rawdump.cxx)
