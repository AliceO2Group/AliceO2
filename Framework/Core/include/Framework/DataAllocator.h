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
#include "Framework/MessageContext.h"
#include "Framework/RootObjectContext.h"
#include "Framework/TMessageSerializer.h"
#include "fairmq/FairMQMessage.h"

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <type_traits>

#include <TClass.h>

namespace o2 {
namespace framework {

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

  // In case no extra argument is provided and the passed type is a POD,
  // the most likely wanted behavior is to create a message with that POD,
  // and so we do.
  template <typename T>
  typename std::enable_if<std::is_pod<T>::value == true, T &>::type
  make(const OutputSpec &spec) {
    DataChunk chunk = newChunk(spec, sizeof(T));
    return *reinterpret_cast<T*>(chunk.data);
  }

  // In case an extra argument is provided, we consider this an array / 
  // collection elements of that type
  template <typename T>
  typename std::enable_if<std::is_pod<T>::value == true, Collection<T>>::type
  make(const OutputSpec &spec, size_t nElements) {
    auto size = nElements*sizeof(T);
    DataChunk chunk = newChunk(spec, size);
    return Collection<T>(chunk.data, nElements);
  }

  /// Use this in case you want to leave the creation
  /// of a TObject to be transmitted to the framework.
  /// @a spec is the specification for the output
  /// @a args is the arguments for the constructor of T
  /// @return a reference to the constructed object. Such an object
  /// will be sent to all the consumers of the output @a spec
  /// once the processing callback completes.
  template <typename T, typename... Args>
  typename std::enable_if<std::is_base_of<TObject, T>::value == true, T&>::type
  make(const OutputSpec &spec, Args... args) {
    auto obj = new T(args...);
    adopt(spec, obj);
    return *obj;
  }

  /// Adopt a TObject in the framework and serialize / send
  /// it to the consumers of @a spec once done.
  void
  adopt(const OutputSpec &spec, TObject*obj);

private:
  std::string matchDataHeader(const OutputSpec &spec, size_t timeframeId);
  FairMQMessagePtr headerMessageFromSpec(OutputSpec const &spec,
                                         std::string const &channel,
                                         o2::Header::SerializationMethod serializationMethod);

  FairMQDevice *mDevice;
  AllowedOutputsMap mAllowedOutputs;
  MessageContext *mContext;
  RootObjectContext *mRootContext;
};

}
}

#endif //FRAMEWORK_DATAALLOCATOR_H
