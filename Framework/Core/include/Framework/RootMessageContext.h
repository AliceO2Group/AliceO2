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
#ifndef O2_FRAMEWORK_ROOTMESSAGECONTEXT_H_
#define O2_FRAMEWORK_ROOTMESSAGECONTEXT_H_

#include "Framework/DispatchControl.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/OutputRoute.h"
#include "Framework/RouteState.h"
#include "Framework/RoutingIndices.h"
#include "Framework/RuntimeError.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/SerializationMethods.h"
#include "Framework/TypeTraits.h"
#include "Framework/MessageContext.h"

#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "MemoryResources/MemoryResources.h"

#include <fairmq/Message.h>
#include <fairmq/Parts.h>

#include <cassert>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include <fairmq/FwdDecls.h>

namespace o2::framework
{

struct Output;

/// RootSerializedObject keeps ownership to an object which can be Root-serialized
/// TODO: this should maybe be a separate header file to avoid including TMessageSerializer
/// in this header file, but we can always change this without affecting to much code.
template <typename T>
class RootSerializedObject : public MessageContext::ContextObject
{
 public:
  // Note: we strictly require the type to implement the ROOT ClassDef interface in order to be
  // able to check for the existence of the dirctionary for this type. Could be dropped if any
  // use case for a type having the dictionary at runtime pops up
  static_assert(has_root_dictionary<T>::value == true, "unconsistent type: needs to implement ROOT ClassDef interface");
  using value_type = T;
  /// default constructor forbidden, object alwasy has to control messages
  RootSerializedObject() = delete;
  /// constructor taking header message by move and creating the object from variadic argument list
  template <typename ContextType, typename... Args>
  RootSerializedObject(ContextType* context, fair::mq::MessagePtr&& headerMsg, RouteIndex routeIndex, Args&&... args)
    : ContextObject(std::forward<fair::mq::MessagePtr>(headerMsg), routeIndex)
  {
    mObject = std::make_unique<value_type>(std::forward<Args>(args)...);
    mPayloadMsg = context->proxy().createOutputMessage(routeIndex);
  }
  ~RootSerializedObject() override = default;

  /// @brief Finalize object and return parts by move
  /// This retrieves the actual message from the vector object and moves it to the parts
  fair::mq::Parts finalize() final
  {
    assert(mParts.Size() == 1);
    TMessageSerializer::Serialize(*mPayloadMsg, mObject.get(), nullptr);
    mParts.AddPart(std::move(mPayloadMsg));
    return ContextObject::finalize();
  }

  operator value_type&()
  {
    return *mObject;
  }

  value_type& get()
  {
    return *mObject;
  }

 private:
  std::unique_ptr<value_type> mObject;
  fair::mq::MessagePtr mPayloadMsg;
};

template <typename T>
struct enable_root_serialization<T, std::enable_if_t<has_root_dictionary<T>::value && is_messageable<T>::value == false>> : std::true_type {
  using object_type = RootSerializedObject<T>;
};

template <typename T>
struct root_serializer<T, std::enable_if_t<has_root_dictionary<T>::value || is_specialization<T, ROOTSerialized>::value == true>> : std::true_type {
  using serializer = TMessageSerializer;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_MESSAGECONTEXT_H_
