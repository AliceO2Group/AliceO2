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
#include "Headers/DataHeader.h"
#include "Framework/DataProcessingHeader.h"

namespace o2
{
namespace framework
{

using DataHeader = o2::header::DataHeader;

InjectorFunction dataSamplingReadoutAdapter(OutputSpec const& spec)
{

  // copied from Common/DataBlock.h
  using DataBlockId = uint64_t;
  struct DataBlockHeaderBase {
    uint32_t blockType;  ///< ID to identify structure type
    uint32_t headerSize; ///< header size in bytes
    uint32_t dataSize;   ///< data size following this structure (until next header, if this is not a toplevel block header)
    DataBlockId id;      ///< id of the block (monotonic increasing sequence)
    uint32_t linkId;     ///< id of link
  };

  return [spec](FairMQDevice& device, FairMQParts& parts, int index) {
    for (size_t i = 0; i < parts.Size() / 2; ++i) {

      auto dbh = reinterpret_cast<DataBlockHeaderBase*>(parts.At(2 * i)->GetData());
      assert(dbh->dataSize == parts.At(2 * i + 1)->GetSize());

      DataHeader dh;
      dh.dataOrigin = spec.origin;
      dh.dataDescription = spec.description;
      dh.subSpecification = dbh->linkId;
      dh.payloadSize = dbh->dataSize;
      dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;

      DataProcessingHeader dph{ dbh->id, 0 };
      o2::header::Stack headerStack{ dh, dph };
      broadcastMessage(device, std::move(headerStack), std::move(parts.At(2 * i + 1)), index);
    }
  };
}

} // namespace framework
} // namespace o2
