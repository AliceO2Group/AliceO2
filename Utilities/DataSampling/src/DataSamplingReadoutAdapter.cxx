// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "DataSampling/DataSamplingReadoutAdapter.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/RawDeviceService.h"
#include "Headers/DataHeader.h"
#include "Framework/DataSpecUtils.h"
#include <atomic>

using namespace o2::framework;

namespace o2::utilities
{

using DataHeader = o2::header::DataHeader;

InjectorFunction dataSamplingReadoutAdapter(OutputSpec const& spec)
{
  return [spec](TimingInfo&, ServiceRegistryRef const& ref, fair::mq::Parts& parts, ChannelRetriever channelRetriever, size_t newTimesliceId, bool& stop) {
    auto *device = ref.get<RawDeviceService>().device();

    for (size_t i = 0; i < parts.Size(); ++i) {

      DataHeader dh;
      ConcreteDataTypeMatcher dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
      dh.dataOrigin = dataType.origin;
      dh.dataDescription = dataType.description;
      dh.subSpecification = DataSpecUtils::getOptionalSubSpec(spec).value_or(0xFF);
      dh.payloadSize = parts.At(i)->GetSize();
      dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;

      DataProcessingHeader dph{newTimesliceId, 0};
      o2::header::Stack headerStack{dh, dph};
      sendOnChannel(*device, std::move(headerStack), std::move(parts.At(i)), spec, channelRetriever);
    }
    return parts.Size() != 0;
  };
}

} // namespace o2::utilities
