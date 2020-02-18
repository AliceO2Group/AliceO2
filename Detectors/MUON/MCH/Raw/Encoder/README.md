<!-- doxy
\page refDetectorsMUONMCHRawEncoder Encoder
/doxy -->

# MCH Raw Data Encoder

Like the decoder, the encoder deals with two formats (Bare and UserLogic), both
of which in two different modes (charge sum and sample mode). Note that only
the chargesum mode is fully useable (but that should not be much of a
    limitation as we're yet to find a real use case for simulating the sample
    mode).

Generation of MCH raw data buffers is a two stage process : first we
build payloads and then only we organize them in (RDH,payload) blocks.

The Encoder is mainly about building *payloads*. How those payloads are ordered
in memory is not the Encoder business. This choice has been made to minimize
the dependency of the Encoder code on the RawDataHeader itself.  To that end
the Encoder is producing data buffers consisting of (header,payload) [blocks](/include/MCHRawEncoder/DataBlock.h), where header is *not* a RDH but a
simpler (and non versioned) struct. Contrary to (RDH,payload) blocks, the
payload can be any size (while in RDH the size is limited to 65536 bytes). The
part of the detector that a payload block references is identified through a
unique `feeId` field which is the `solarId` in MCH case.

Methods are then provided to massage the (header,payload) buffers to produce
realistic (RDH,payload) buffers that closely ressemble real raw data.

## createEncoder&lt;FORMAT,CHARGESUM>

To creation of a MCH Encoder is handled through a template function.
For instance, to create an Encoder that builds raw data in BareFormat and
ChargeSumMode, use :

```.cpp
#include "MCHRawEncoder/Encoder.h"

auto encoder = o2::mch::raw::Encoder<BareFormat,ChargeSumMode>();

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
