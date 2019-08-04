// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cstring>

#include "DataFlow/TimeframeReaderDevice.h"
#include "DataFlow/TimeframeParser.h"
#include "Headers/SubframeMetadata.h"
#include "Headers/DataHeader.h"
#include <options/FairMQProgOptions.h>

using DataHeader = o2::header::DataHeader;

namespace o2
{
namespace data_flow
{

TimeframeReaderDevice::TimeframeReaderDevice()
  : O2Device{}, mOutChannelName{}, mFile{}
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
  auto addPartFn = [this](FairMQParts& parts, char* buffer, size_t size) {
    parts.AddPart(this->NewMessage(
      buffer,
      size,
      [](void* data, void* hint) { delete[](char*) data; },
      nullptr));
  };
  auto sendFn = [this](FairMQParts& parts) { this->Send(parts, this->mOutChannelName); };

  // FIXME: For the moment we support a single file. This should really be a glob. We
  //        should also have a strategy for watching directories.
  std::vector<std::string> files;
  files.push_back(mInFileName);
  for (auto&& fn : files) {
    mFile.open(fn, std::ofstream::in | std::ofstream::binary);
    try {
      streamTimeframe(mFile,
                      addPartFn,
                      sendFn);
    } catch (std::runtime_error& e) {
      LOG(ERROR) << e.what() << "\n";
    }
    mSeen.push_back(fn);
  }

  return false;
}

} // namespace data_flow
} // namespace o2
