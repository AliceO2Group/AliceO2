<!-- doxy
\page refDetectorsMUONMCHRawEncoderPayload Payload Encoder
/doxy -->

The `EncoderPayload` library is building raw data buffer for *data payloads*.
How those payloads are then ordered exactly in memory / on file is not the
library business. This choice has been made to minimize the dependency of the
Encoder code on the RawDataHeader itself.  To that end the Encoder is producing
data buffers consisting of (header,payload)
[blocks](/include/MCHRawEncoderPayload/DataBlock.h), where header is *not* a
RDH but a simpler (and non versioned) struct. Contrary to (RDH,payload) blocks,
the payload can be any size (while in RDH the size is limited to 65536 bytes).
The part of the detector that a payload block references is identified through
the `solarId` field.

## createPayloadEncoder

To creation of a MCH PayloadEncoder is handled through a template function.
For instance, to create an Encoder that builds raw data in BareFormat and
ChargeSumMode, use :

```.cpp
#include "MCHRawEncoderPayload/PayloadEncoder.h"

auto encoder = o2::mch::raw::createPayloadEncoder<BareFormat,ChargeSumMode>();

```

Then for each interaction record (aka orbit,bunch crossing pair) you tell
the encoder to start a new Heartbeat Frame, fill it with all the data
of the channels and then extract the memory buffer.

```.cpp
vector<uint8_t> buffer;
for ( loop over interactions )
  encoder->startHeartbeatFrame(orbit,bc);
  for ( loop over digits ) {
    DsElecId elecId = ... // get electronic dual sampa id from digit
      int sampaChannelId = ... // get sampa channel id from digits
      vector<SampaClusters> clusters = ... // create sampa clusters from digits
      encoder->addChannelData(elecId,sampaChannelId,clusters);
  }
encoder->moveToBuffer(buffer);
}
```

At this stage the buffer contains (header,payload) blocks for all the
interactions you looped over. But it is *not* yet a valid raw data, as it lacks
RDHs (and has mch-specific headers instead). 

The conversion from payload buffers to RDH-base buffers is basically done using
the `o2::mch::raw::PayloadPaginator` class (internally using the `o2::raw::RawFileWriter` framework
class).  A typical usage example is in the [digit2raw](Digit/) program.

## Advanced usage

The `createPayloadEncoder` actually takes one parameter, a function that
converts a solarId into a FeeLinkId (aka a Solar2FeeLinkMapper).  The default
value should fit the electronic mapping of the actual detector and thus should
be fine in most cases. For special use cases a different mapper can be provided
though.
