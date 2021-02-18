// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "DataSampling/DataSamplingReadoutAdapter.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include "Framework/DataSpecUtils.h"
#include <atomic>

using namespace o2::framework;

namespace o2::utilities
{

using DataHeader = o2::header::DataHeader;

static std::atomic<unsigned int> blockId = 0;

InjectorFunction dataSamplingReadoutAdapter(OutputSpec const& spec)
{
  return [spec](FairMQDevice& device, FairMQParts& parts, ChannelRetriever channelRetriever) {
    for (size_t i = 0; i < parts.Size() / 2; ++i) {

      DataHeader dh;
      ConcreteDataTypeMatcher dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
      dh.dataOrigin = dataType.origin;
      dh.dataDescription = dataType.description;
      dh.subSpecification = DataSpecUtils::getOptionalSubSpec(spec).value_or(0xFF);
      dh.payloadSize = parts.At(2 * i + 1)->GetSize();
      dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;

      DataProcessingHeader dph{++blockId, 0};
      o2::header::Stack headerStack{dh, dph};
      sendOnChannel(device, std::move(headerStack), std::move(parts.At(2 * i + 1)), spec, channelRetriever);
    }
  };
}

} // namespace o2::utilities
