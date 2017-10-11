# Data Distribution


Data Distribution O2Devices implementing SubTimeFrame building and data distribution tasks.

## Demonstrator chain

The chain starts with a `ReadoutDevice` (currently emulated) which produces HBFrames. Next in chain is `SubTimeFrameBuilderDevice`, which collects HBFrames belonging to the same SubTimeFrames. The final device in the chain is `SubTimeFrameTransporterDevice`, which is responsible for transporting the data to the EPN (to be implemented) and free the resources of the transported STF (readout buffers, shared memory allocated in the local processing, etc).

## Interfaces and data model

Readout process(es) send collections of STFrames when they become available (e.g. when a CRU superpage is completely filled). The exchange is implemented in the `O2SubTimeFrameLinkData` class. `SubTimeFrameBuilder` device uses an object of `O2SubTimeFrame` class to track all HBFrames of the same STF. When the STF is built, it's sent to the next device in the chain using objects implementing visitor pattern (e.g. a pair of `InterleavedHdrDataSerializer` and `InterleavedHdrDataDeserializer` objects). These objects enumerate all data and headers of the STF, collecting all FairMQ messages for sending. Since all STF data is allocated in SHM the STF moving process is zero-copy.

## Running

O2Device channel configuration can be found in `readout-flp-chain.json`. The current configuration uses shared memory channels for all devices on the FLPs. To run the chain, please refer to `startReadoutFLPChain.sh` script.
Options:
 - `NUM_CRUS`: the number of CRU readout processes. Can be set to 1 or 2
 - `DMA_STREAMS_PER_CRU`: this parameter determines how many independent HBFrame streams is being produced per CRU.
 - `DMA_STREAM_BITS_PER_S`: average data rate of each DMA stream
 - `DATA_REGION_SIZE`: Size of Shared Memory region used by a single CRU (decrease for small-scale testing)
 - `SUPERPAGE_SIZE`: (CRU emulator internal) superpage size
 - `DMA_CHUNK_SIZE`: (CRU emulator internal) DMA engine native block size. When set to zero the CRU emulator always produces contiguous HBFrames.
