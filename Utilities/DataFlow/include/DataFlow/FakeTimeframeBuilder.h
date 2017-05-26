#ifndef DATAFLOW_FAKETIMEFRAMEBUILDER_H_
#define DATAFLOW_FAKETIMEFRAMEBUILDER_H_

#include "Headers/DataHeader.h"
#include <vector>
#include <memory>
#include <functional>

namespace o2 { namespace DataFlow {

struct FakeTimeframeSpec {
  const char *origin;
  const char *dataDescription;
  std::function<void(char *, size_t)> bufferFiller;
  size_t bufferSize;
};

/** Generate a timeframe from the provided specification
  */
std::unique_ptr<char[]> fakeTimeframeGenerator(std::vector<FakeTimeframeSpec> &specs, std::size_t &totalSize);

}}
#endif /* DATAFLOW_FAKETIMEFRAMEBUILDER_H_ */
