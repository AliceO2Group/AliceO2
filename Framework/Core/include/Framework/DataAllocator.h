// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAALLOCATOR_H
#define FRAMEWORK_DATAALLOCATOR_H

#include <fairmq/FairMQDevice.h>
#include "Headers/DataHeader.h"
#include "Framework/OutputRoute.h"
#include "Framework/DataChunk.h"
#include "Framework/Collection.h"

#include <vector>
#include <map>
#include <string>
#include <utility>

class TClonesArray;

namespace o2 {
namespace framework {

class MessageContext;
class RootObjectContext;

/// This allocator is responsible to make sure that the messages created match
/// the provided spec and that depending on how many pipelined reader we
/// have, messages get created on the channel for the reader of the current
/// timeframe.
class DataAllocator
{
public:
  using AllowedOutputsMap = std::vector<OutputRoute>;
  using DataOrigin = o2::Header::DataOrigin;
  using DataDescription = o2::Header::DataDescription;
  using SubSpecificationType = o2::Header::DataHeader::SubSpecificationType;

  DataAllocator(FairMQDevice *device,
                MessageContext *context,
                RootObjectContext *rootContext,
                const AllowedOutputsMap &outputs);
  DataChunk newChunk(const OutputSpec &, size_t);
  DataChunk adoptChunk(const OutputSpec &, char *, size_t, fairmq_free_fn*, void *);
  TClonesArray &newTClonesArray(const OutputSpec &, const char *, size_t);

  template <class T>
  Collection<T> newCollectionChunk(const OutputSpec &spec, size_t nElements) {
    static_assert(std::is_pod<T>::value == true, "Type must be a PoD");
    auto size = nElements*sizeof(T);
    DataChunk chunk = newChunk(spec, size);
    return Collection<T>(chunk.data, nElements);
  }

private:
  std::string matchDataHeader(const OutputSpec &spec, size_t timeframeId);
  FairMQDevice *mDevice;
  AllowedOutputsMap mAllowedOutputs;
  MessageContext *mContext;
  RootObjectContext *mRootContext;
};

}
}

#endif //FRAMEWORK_DATAALLOCATOR_H
