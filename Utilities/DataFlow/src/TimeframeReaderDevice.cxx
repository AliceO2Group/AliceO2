#include <cstring>

#include "DataFlow/TimeframeReaderDevice.h"
#include "DataFlow/TimeframeParser.h"
#include "Headers/SubframeMetadata.h"
#include "Headers/DataHeader.h"
#include <options/FairMQProgOptions.h>

using DataHeader = o2::Header::DataHeader;

namespace o2 { namespace DataFlow {

TimeframeReaderDevice::TimeframeReaderDevice()
  : O2Device{}
  , mOutChannelName{}
  , mFile{}
{
}

void TimeframeReaderDevice::InitTask()
{
  mOutChannelName = GetConfig()->GetValue<std::string>(OptionKeyOutputChannelName);
  mInFileName = GetConfig()->GetValue<std::string>(OptionKeyInputFileName);
  mSeen.clear();
}

bool TimeframeReaderDevice::ConditionalRun()
{
  auto addPartFn = [this](FairMQParts &parts, char *buffer, size_t size) {
        parts.AddPart(this->NewMessage(buffer,
                                       size,
                                       [](void* data, void* hint) { delete[] (char*)data; },
                                       nullptr));
  };
  auto sendFn = [this](FairMQParts &parts) {this->Send(parts, this->mOutChannelName);};

  // FIXME: For the moment we support a single file. This should really be a glob. We
  //        should also have a strategy for watching directories.
  std::vector<std::string> files;
  files.push_back(mInFileName);
  for (auto &&fn : files) {
    mFile.open(fn, std::ofstream::in | std::ofstream::binary);
    try {
      streamTimeframe(mFile,
                      addPartFn,
                      sendFn);
    } catch(std::runtime_error &e) {
      LOG(ERROR) << e.what() << "\n";
    }
    mSeen.push_back(fn);
  }

  return false;
}

}} // namespace o2::DataFlow
