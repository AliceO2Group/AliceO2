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

#include "Framework/ReadoutAdapter.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataSpecUtils.h"
#include "Headers/DataHeader.h"

namespace o2
{
namespace framework
{

using DataHeader = o2::header::DataHeader;

InjectorFunction readoutAdapter(OutputSpec const& spec)
{
  auto counter = std::make_shared<uint64_t>(0);

  return [spec, counter](TimingInfo&, fair::mq::Device& device, fair::mq::Parts& parts, ChannelRetriever channelRetriever) {
    for (size_t i = 0; i < parts.Size(); ++i) {
      DataHeader dh;
      // FIXME: this will have to change and extract the actual subspec from
      //        the data.
      ConcreteDataMatcher concrete = DataSpecUtils::asConcreteDataMatcher(spec);
      dh.dataOrigin = concrete.origin;
      dh.dataDescription = concrete.description;
      dh.subSpecification = concrete.subSpec;
      dh.payloadSize = parts.At(i)->GetSize();
      dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;

      DataProcessingHeader dph{*counter, 0};
      (*counter) += 1UL;
      o2::header::Stack headerStack{dh, dph};
      sendOnChannel(device, std::move(headerStack), std::move(parts.At(i)), spec, channelRetriever);
    }
  };
}

} // namespace framework
} // namespace o2
