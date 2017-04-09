#ifndef SUBFRAMEMETADATA_H
#define SUBFRAMEMETADATA_H

namespace o2 {
namespace DataFlow {

struct SubframeMetadata
{
  // TODO: replace with timestamp struct
  // IDEA: not timeframeID because can be calculcated with helper function
  // QUESTION: isn't the duration set to ~22ms?
  uint64_t startTime = ~(uint64_t)0;
  uint64_t duration = ~(uint64_t)0;

  //further meta data to be added

  // putting data specific to FLP origin
  int      flpIndex;
};

// Helper function to derive the timeframe id from the actual timestamp.
// Timestamp is in nanoseconds. Each Timeframe is ~22ms i.e. 2^17 nanoseconds,
// so we can get a unique id by dividing by the timeframe period and masking 
// the lower 16 bits. Overlaps will only happen every ~ 22 minutes.
constexpr uint16_t
timeframeIdFromTimestamp(uint64_t timestamp, uint64_t timeFrameDuration) {
  return (timestamp / timeFrameDuration) & 0xffff;
}

// A Mockup class to describe some TPC-like payload
struct TPCTestCluster {
  float x = 0.f;
  float y = 0.f;
  float z = 1.5f;
  float q = 0.;
  uint64_t timeStamp; // the time this thing was digitized/recorded
};

struct TPCTestPayload {
  std::vector<TPCTestCluster> clusters;
};

// a mockup class to describe some "ITS" payload
struct ITSRawData {
  float x = -1.;
  float y = 1.;
  uint64_t timeStamp;
};


} // end namespace DataFlow
} // end namespace AliceO2


#endif
