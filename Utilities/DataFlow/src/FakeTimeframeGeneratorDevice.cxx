#include <cstring>

#include "DataFlow/FakeTimeframeGeneratorDevice.h"
#include "DataFlow/FakeTimeframeBuilder.h"
#include "DataFlow/TimeframeParser.h"
#include "Headers/SubframeMetadata.h"
#include "Headers/DataHeader.h"
#include <options/FairMQProgOptions.h>
#include <vector>

using DataHeader = o2::Header::DataHeader;

namespace {
struct OneShotReadBuf : public std::streambuf
{
    OneShotReadBuf(char* s, std::size_t n)
    {
        setg(s, s, s + n);
    }
};
}
namespace o2 { namespace DataFlow {

FakeTimeframeGeneratorDevice::FakeTimeframeGeneratorDevice()
  : O2Device{}
  , mOutChannelName{}
  , mMaxTimeframes{}
  , mTimeframeCount{0}
{
}

void FakeTimeframeGeneratorDevice::InitTask()
{
  mOutChannelName = GetConfig()->GetValue<std::string>(OptionKeyOutputChannelName);
  mMaxTimeframes = GetConfig()->GetValue<size_t>(OptionKeyMaxTimeframes);
}

bool FakeTimeframeGeneratorDevice::ConditionalRun()
{
  auto addPartFn = [this](FairMQParts &parts, char *buffer, size_t size) {
        parts.AddPart(this->NewMessage(buffer,
                                       size,
                                       [](void* data, void* hint) { delete[] (char*)data; },
                                       nullptr));
  };
  auto sendFn = [this](FairMQParts &parts) {this->Send(parts, this->mOutChannelName);};
  auto zeroFiller = [](char *b, size_t s) {memset(b, 0, s);};

  std::vector<o2::DataFlow::FakeTimeframeSpec> specs = {
    {
      .origin = "TPC",
      .dataDescription = "CLUSTERS",
      .bufferFiller = zeroFiller,
      .bufferSize = 1000
    },
    {
      .origin = "ITS",
      .dataDescription = "CLUSTERS",
      .bufferFiller = zeroFiller,
      .bufferSize = 500
    }
  };

  try {
    size_t totalSize;
    auto buffer = fakeTimeframeGenerator(specs, totalSize);
    OneShotReadBuf osrb(buffer.get(), totalSize);
    std::istream s(&osrb);

    streamTimeframe(s,
                    addPartFn,
                    sendFn);
  } catch(std::runtime_error &e) {
    LOG(ERROR) << e.what() << "\n";
  }

  mTimeframeCount++;

  if (mTimeframeCount < mMaxTimeframes) {
    return true;
  }
  return false;
}

}} // namespace o2::DataFlow
