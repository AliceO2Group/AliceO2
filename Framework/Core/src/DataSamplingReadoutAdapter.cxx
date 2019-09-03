// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataSamplingReadoutAdapter.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include "Framework/DataSpecUtils.h"

#include <Common/DataBlock.h>

namespace o2::framework
{

using DataHeader = o2::header::DataHeader;

InjectorFunction dataSamplingReadoutAdapter(OutputSpec const& spec)
{
  return [spec](FairMQDevice& device, FairMQParts& parts, int index) {
    for (size_t i = 0; i < parts.Size() / 2; ++i) {

      auto dbh = reinterpret_cast<DataBlockHeaderBase*>(parts.At(2 * i)->GetData());
      assert(dbh->dataSize == parts.At(2 * i + 1)->GetSize());

      DataHeader dh;
      ConcreteDataTypeMatcher dataType = DataSpecUtils::asConcreteDataTypeMatcher(spec);
      dh.dataOrigin = dataType.origin;
      dh.dataDescription = dataType.description;
      dh.subSpecification = DataSpecUtils::getOptionalSubSpec(spec).value_or(dbh->linkId);
      dh.payloadSize = dbh->dataSize;
      dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;

      DataProcessingHeader dph{dbh->blockId, 0};
      o2::header::Stack headerStack{dh, dph};
      broadcastMessage(device, std::move(headerStack), std::move(parts.At(2 * i + 1)), index);
    }
  };
}

} // namespace o2::framework
