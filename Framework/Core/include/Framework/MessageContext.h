// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_MESSAGECONTEXT_H
#define FRAMEWORK_MESSAGECONTEXT_H

#include "Framework/FairMQDeviceProxy.h"
#include "Framework/TypeTraits.h"

#include <fairmq/FairMQMessage.h>
#include <fairmq/FairMQParts.h>

#include <vector>
#include <cassert>
#include <string>
#include <type_traits>

class FairMQDevice;

namespace o2
{
namespace framework
{

class MessageContext {
 public:
  MessageContext(FairMQDeviceProxy proxy)
    : mProxy{ proxy }
  {
  }

  // thhis is the virtual interface for context objects
  class ContextObject
  {
   public:
    ContextObject() = default;
    ContextObject(FairMQMessagePtr&& headerMsg, FairMQMessagePtr&& payloadMsg, const std::string& bindingChannel)
      : parts{}, channel{ bindingChannel }
    {
      parts.AddPart(std::move(headerMsg));
      parts.AddPart(std::move(payloadMsg));
    }
    virtual ~ContextObject() = default;

    // TODO: we keep this public for the moment to keep the current interface
    // the send-handler loops over all objects and can acces the message parts
    // directly, needs to be changed when the context is generalized, then the
    // information needs to be stored in the derived class
    FairMQParts parts;
    std::string channel;
  };

  /// TrivialObject handles a message object
  class TrivialObject : public ContextObject
  {
   public:
    /// default contructor forbidden, object always has to control message instances
    TrivialObject() = delete;
    /// constructor consuming the header and payload messages for a given channel by move
    template <typename ContextType>
    TrivialObject(ContextType*, FairMQMessagePtr&& headerMsg, FairMQMessagePtr&& payloadMsg, const std::string& bindingChannel)
      : ContextObject(std::forward<FairMQMessagePtr>(headerMsg), std::forward<FairMQMessagePtr>(payloadMsg), bindingChannel)
    {
    }
    /// constructor taking header message by move and creating the paypload message
    template <typename ContextType, typename... Args>
    TrivialObject(ContextType* context, FairMQMessagePtr&& headerMsg, const std::string& bindingChannel, int index, Args... args)
      : ContextObject(std::forward<FairMQMessagePtr>(headerMsg), context->createMessage(bindingChannel, index, std::forward<Args>(args)...), bindingChannel)
    {
    }
    ~TrivialObject() override = default;

    auto* data()
    {
      assert(parts.Size() == 2);
      return parts[1].GetData();
    }
  };

  // SpanObject creates a trivial binary object for an array of elements of
  // type T and holds a span over the elements
  template <typename T>
  class SpanObject : public ContextObject
  {
   public:
    static_assert(is_messageable<T>::value == true, "unconsistent type");
    using value_type = gsl::span<T>;
    /// default constructor forbidden, object alwasy has to control messages
    SpanObject() = delete;
    /// constructor taking header message by move and creating the payload message for the span
    template <typename ContextType>
    SpanObject(ContextType* context, FairMQMessagePtr&& headerMsg, const std::string& bindingChannel, int index, size_t nElements)
    {
      // create the span object for the memory of the payload message
      // TODO: we probably also want to check consistency of the header message, i.e. payloadSize member
      auto payloadMsg = context->createMessage(bindingChannel, index, nElements * sizeof(T));
      mValue = value_type(reinterpret_cast<T*>(payloadMsg->GetData()), nElements);
      parts.AddPart(std::move(headerMsg));
      parts.AddPart(std::move(payloadMsg));
      channel = bindingChannel;
    }
    ~SpanObject() override = default;

    operator value_type&()
    {
      return mValue;
    }

    value_type& get()
    {
      return mValue;
    }

   private:
    value_type mValue;
  };
  using Messages = std::vector<std::unique_ptr<ContextObject>>;

  template <typename T, typename... Args>
  auto& add(Args&&... args)
  {
    static_assert(std::is_base_of<ContextObject, T>::value == true, "type must inherit ContextObject interface");
    mMessages.push_back(std::move(std::make_unique<T>(this, std::forward<Args>(args)...)));
    return *dynamic_cast<T*>(mMessages.back().get());
  }

  Messages::iterator begin()
  {
    return mMessages.begin();
  }

  Messages::iterator end()
  {
    return mMessages.end();
  }

  size_t size()
  {
    return mMessages.size();
  }

  /// Prepares the context to create messages for the given timeslice. This
  /// expects that the previous context was already sent and can be completely
  /// discarded.
  void clear()
  {
    // Verify that everything has been sent on clear.
    for (auto &m : mMessages) {
      assert(m->parts.Size() == 0);
    }
    mMessages.clear();
  }

  FairMQDeviceProxy& proxy()
  {
    return mProxy;
  }

  /// call the proxy to create a message of the specified size
  /// we don't implement in the header to avoid including the FairMQDevice header here
  /// that's why the different versions need to be implemented as individual functions
  // FIXME: can that be const?
  FairMQMessagePtr createMessage(const std::string& channel, int index, size_t size);
  FairMQMessagePtr createMessage(const std::string& channel, int index, void* data, size_t size, fairmq_free_fn* ffn, void* hint);

 private:
  FairMQDeviceProxy mProxy;
  Messages mMessages;
};
} // namespace framework
} // namespace o2
#endif // FRAMEWORK_MESSAGECONTEXT_H
