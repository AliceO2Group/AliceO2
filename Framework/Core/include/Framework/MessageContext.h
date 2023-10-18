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
#ifndef O2_FRAMEWORK_MESSAGECONTEXT_H_
#define O2_FRAMEWORK_MESSAGECONTEXT_H_

#include "Framework/DispatchControl.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/OutputRoute.h"
#include "Framework/RouteState.h"
#include "Framework/RoutingIndices.h"
#include "Framework/RuntimeError.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/TypeTraits.h"

#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "MemoryResources/MemoryResources.h"

#include <fairmq/Message.h>
#include <fairmq/Parts.h>

#include <cassert>
#include <functional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <fairmq/FwdDecls.h>

namespace o2::framework
{

template <typename T, typename = void>
struct enable_root_serialization : std::false_type {
  using debug_type = T;
};

template <typename T, typename = void>
struct root_serializer : std::false_type {
};

struct Output;

class MessageContext
{
 public:
  constexpr static ServiceKind service_kind = ServiceKind::Stream;

  // so far we are only using one instance per named channel
  static constexpr int DefaultChannelIndex = 0;

  MessageContext(FairMQDeviceProxy& proxy)
    : mProxy{proxy}
  {
  }

  MessageContext(FairMQDeviceProxy& proxy, DispatchControl&& dispatcher)
    : mProxy{proxy}, mDispatchControl{dispatcher}
  {
  }

  void init(DispatchControl&& dispatcher)
  {
    mDispatchControl = dispatcher;
  }

  // this is the virtual interface for context objects
  class ContextObject
  {
   public:
    ContextObject() = delete;
    ContextObject(fair::mq::MessagePtr&& headerMsg, fair::mq::MessagePtr&& payloadMsg, RouteIndex routeIndex)
      : mParts{}, mRouteIndex{routeIndex}
    {
      mParts.AddPart(std::move(headerMsg));
      mParts.AddPart(std::move(payloadMsg));
    }

    ContextObject(fair::mq::MessagePtr&& headerMsg, RouteIndex routeIndex)
      : mParts{}, mRouteIndex{routeIndex}
    {
      mParts.AddPart(std::move(headerMsg));
    }

    virtual ~ContextObject() = default;

    /// @brief Finalize the object and return the parts by move
    /// This is the default method and can be overloaded by other implmentations to carry out other
    /// tasks before returning the parts objects
    virtual fair::mq::Parts finalize()
    {
      fair::mq::Parts parts = std::move(mParts);
      assert(parts.Size() == 2);
      auto* header = o2::header::get<o2::header::DataHeader*>(parts.At(0)->GetData());
      if (header == nullptr) {
        throw std::logic_error("No valid header message found");
      } else {
        // o2::header::get returns const pointer, but here we can change the message
        const_cast<o2::header::DataHeader*>(header)->payloadSize = parts.At(1)->GetSize();
      }
      // return value optimization returns by move
      return parts;
    }

    /// @brief return the channel name
    [[nodiscard]] RouteIndex route() const
    {
      return mRouteIndex;
    }

    [[nodiscard]] bool empty() const
    {
      return mParts.Size() == 0;
    }

    o2::header::DataHeader const* header()
    {
      // we would expect this function to be const but the fair::mq::Parts API does not allow this
      if (empty() || mParts.At(0) == nullptr) {
        return nullptr;
      }
      return o2::header::get<o2::header::DataHeader*>(mParts.At(0)->GetData());
    }

    o2::framework::DataProcessingHeader const* dataProcessingHeader()
    {
      if (empty() || mParts.At(0) == nullptr) {
        return nullptr;
      }
      return o2::header::get<o2::framework::DataProcessingHeader*>(mParts.At(0)->GetData());
    }

    o2::header::Stack const* headerStack()
    {
      // we would expect this function to be const but the fair::mq::Parts API does not allow this
      if (empty() || mParts.At(0) == nullptr) {
        return nullptr;
      }
      return o2::header::get<o2::header::DataHeader*>(mParts.At(0)->GetData()) ? reinterpret_cast<o2::header::Stack*>(mParts.At(0)->GetData()) : nullptr;
    }

   protected:
    fair::mq::Parts mParts;
    RouteIndex mRouteIndex{-1};
  };

  /// TrivialObject handles a message object
  class TrivialObject : public ContextObject
  {
   public:
    /// default contructor forbidden, object always has to control message instances
    TrivialObject() = delete;
    /// constructor consuming the header and payload messages for a given channel by move
    template <typename ContextType>
    TrivialObject(ContextType* context, fair::mq::MessagePtr&& headerMsg, fair::mq::MessagePtr&& payloadMsg, RouteIndex routeIndex)
      : ContextObject(std::forward<fair::mq::MessagePtr>(headerMsg), std::forward<fair::mq::MessagePtr>(payloadMsg), routeIndex)
    {
    }
    /// constructor taking header message by move and creating the paypload message
    template <typename ContextType, typename... Args>
    TrivialObject(ContextType* context, fair::mq::MessagePtr&& headerMsg, RouteIndex routeIndex, int index, Args... args)
      : ContextObject(std::forward<fair::mq::MessagePtr>(headerMsg), context->createMessage(routeIndex, index, std::forward<Args>(args)...), routeIndex)
    {
    }
    ~TrivialObject() override = default;

    auto* data()
    {
      assert(mParts.Size() == 2);
      return mParts[1].GetData();
    }
  };

  // A memory resource which can force a minimum alignment, so that
  // the whole polymorphic allocator business is happy...
  class AlignedMemoryResource : public pmr::FairMQMemoryResource
  {
   public:
    AlignedMemoryResource(fair::mq::MemoryResource* other)
      : mUpstream(other)
    {
    }

    AlignedMemoryResource(AlignedMemoryResource const& other)
      : mUpstream(other.mUpstream)
    {
    }

    bool isValid()
    {
      return mUpstream != nullptr;
    }
    fair::mq::MessagePtr getMessage(void* p) override
    {
      return mUpstream->getMessage(p);
    }

    void* setMessage(fair::mq::MessagePtr fmm) override
    {
      return mUpstream->setMessage(std::move(fmm));
    }

    fair::mq::TransportFactory* getTransportFactory() noexcept override
    {
      return mUpstream->getTransportFactory();
    }

    [[nodiscard]] size_t getNumberOfMessages() const noexcept override
    {
      return mUpstream->getNumberOfMessages();
    }

   protected:
    void* do_allocate(size_t bytes, size_t alignment) override
    {
      return mUpstream->allocate(bytes, alignment < 64 ? 64 : alignment);
    }

    void do_deallocate(void* p, size_t bytes, size_t alignment) override
    {
      return mUpstream->deallocate(p, bytes, alignment < 64 ? 64 : alignment);
    }

    [[nodiscard]] bool do_is_equal(const pmr::memory_resource& other) const noexcept override
    {
      return this == &other;
    }

   private:
    fair::mq::MemoryResource* mUpstream = nullptr;
  };

  /// ContainerRefObject handles a message object holding an instance of type T
  /// The allocator type is required to be o2::pmr::polymorphic_allocator
  /// can not adopt an existing message, because the polymorphic_allocator will call type constructor,
  /// so this works only with new messages
  /// FIXME: not sure if we want to have this for all container types
  template <typename T>
  class ContainerRefObject : public ContextObject
  {
   public:
    using value_type = typename T::value_type;
    using return_type = T;
    using buffer_type = return_type;
    static_assert(std::is_base_of<o2::pmr::polymorphic_allocator<value_type>, typename T::allocator_type>::value, "container must have polymorphic allocator");
    /// default contructor forbidden, object always has to control message instances
    ContainerRefObject() = delete;
    /// constructor taking header message by move and creating the paypload message
    template <typename ContextType, typename... Args>
    ContainerRefObject(ContextType* context, fair::mq::MessagePtr&& headerMsg, RouteIndex routeIndex, int index, Args&&... args)
      : ContextObject(std::forward<fair::mq::MessagePtr>(headerMsg), routeIndex),
        // the transport factory
        mFactory{context->proxy().getOutputTransport(routeIndex)},
        // the memory resource takes ownership of the message
        mResource{mFactory ? AlignedMemoryResource(mFactory->GetMemoryResource()) : AlignedMemoryResource(nullptr)},
        // create the vector with apropriate underlying memory resource for the message
        mData{std::forward<Args>(args)..., pmr::polymorphic_allocator<value_type>(&mResource)}
    {
      // FIXME: drop this repeated check and make sure at initial setup of devices that everything is fine
      // introduce error policy
      if (mFactory == nullptr) {
        throw runtime_error_f("failed to get transport factory for route %d", routeIndex);
      }
      if (mResource.isValid() == false) {
        throw runtime_error_f("no memory resource for channel %d", routeIndex);
      }
    }
    ~ContainerRefObject() override = default;

    /// @brief Finalize object and return parts by move
    /// This retrieves the actual message from the vector object and moves it to the parts
    fair::mq::Parts finalize() final
    {
      assert(mParts.Size() == 1);
      auto payloadMsg = o2::pmr::getMessage(std::move(mData));
      mParts.AddPart(std::move(payloadMsg));
      return ContextObject::finalize();
    }

    /// @brief return reference to the handled vector object
    operator return_type&()
    {
      return mData;
    }

    /// @brief return reference to the handled vector object
    return_type& get()
    {
      return mData;
    }

    /// @brief return data pointer of the handled vector object
    value_type* data()
    {
      return mData.data();
    }

   private:
    fair::mq::TransportFactory* mFactory = nullptr; /// pointer to transport factory
    AlignedMemoryResource mResource;                /// message resource
    buffer_type mData;                              /// the data buffer
  };

  /// VectorObject handles a message object holding std::vector with polymorphic_allocator
  /// can not adopt an existing message, because the polymorphic_allocator will call the element constructor,
  /// so this works only with new messages
  template <typename T, typename _BASE = ContainerRefObject<std::vector<T, o2::pmr::polymorphic_allocator<T>>>>
  class VectorObject : public _BASE
  {
   public:
    template <typename... Args>
    VectorObject(Args&&... args) : _BASE(std::forward<Args>(args)...)
    {
    }
  };

  // SpanObject creates a trivial binary object for an array of elements of
  // type T and holds a span over the elements
  // FIXME: probably obsolete after introducing of vector with polymorphic_allocator
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
    SpanObject(ContextType* context, fair::mq::MessagePtr&& headerMsg, RouteIndex routeIndex, int index, size_t nElements)
      : ContextObject(std::forward<fair::mq::MessagePtr>(headerMsg), routeIndex)
    {
      // create the span object for the memory of the payload message
      // TODO: we probably also want to check consistency of the header message, i.e. payloadSize member
      auto payloadMsg = context->createMessage(routeIndex, index, nElements * sizeof(T));
      mValue = value_type(reinterpret_cast<T*>(payloadMsg->GetData()), nElements);
      assert(mParts.Size() == 1);
      mParts.AddPart(std::move(payloadMsg));
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

  /// @class ScopeHook A special deleter to handle object going out of scope
  /// This object is used together with the wrapper @a ContextObjectScope which
  /// is a unique object returned to the called. When this scope handler goes out of
  /// scope and is going to be deleted, the handled object is scheduled in the context,
  /// or simply deleted if no context is available.
  template <typename T, typename BASE = std::default_delete<T>>
  class ScopeHook : public BASE
  {
   public:
    using base = std::default_delete<T>;
    using self_type = ScopeHook<T>;
    ScopeHook() = default;
    ScopeHook(MessageContext* context)
      : mContext(context)
    {
    }
    ~ScopeHook() = default;

    // forbid assignment operator to prohibid changing the Deleter
    // resource control property once used in the unique_ptr
    self_type& operator=(const self_type&) = delete;

    void operator()(T* ptr) const
    {
      if (!mContext) {
        // TODO: decide whether this is an error or not
        // can also check if the standard constructor can be dropped to make sure that
        // the ScopeHook is always set up with a context
        throw runtime_error("No context available to schedule the context object");
        return base::operator()(ptr);
      }
      // keep the object alive and add to message list of the context
      mContext->schedule(Messages::value_type(ptr));
    }

   private:
    MessageContext* mContext = nullptr;
  };

  template <typename T>
  using ContextObjectScope = std::unique_ptr<T, ScopeHook<T>>;

  /// Create the specified context object from the variadic arguments and add to message list.
  /// The context object is owned be the context and returned by reference.
  /// The context object type is specified as template argument, each context object implementation
  /// must derive from the ContextObject interface.
  /// TODO: rename to make_ref
  template <typename T, typename... Args>
  auto& add(Args&&... args)
  {
    mMessages.push_back(std::move(make<T>(std::forward<Args>(args)...)));
    // return a reference to the element in the vector of unique pointers
    return *dynamic_cast<T*>(mMessages.back().get());
  }

  /// Create the specified context object from the variadic arguments as a unique pointer of the context
  /// object base class.
  /// The context object type is specified as template argument, each context object implementation
  /// must derive from the ContextObject interface.
  template <typename T, typename... Args>
  Messages::value_type make(Args&&... args)
  {
    static_assert(std::is_base_of<ContextObject, T>::value == true, "type must inherit ContextObject interface");
    return std::make_unique<T>(this, std::forward<Args>(args)...);
  }

  /// Create scope handler managing the specified context object.
  /// The context object is created from the variadic arguments and is owned by the scope handler.
  /// If the handler goes out of scope, the object is scheduled in the context, either added to the
  /// list of messages or directly sent via the optional callback.
  template <typename T, typename... Args>
  ContextObjectScope<T> make_scoped(Args&&... args)
  {
    ContextObjectScope<T> scope(dynamic_cast<T*>(make<T>(std::forward<Args>(args)...).release()), ScopeHook<T>(this));
    return scope;
  }

  /// Schedule a context object for sending.
  /// The object is considered complete at this point and is sent directly through the dispatcher callback
  /// of the context if initialized.
  void schedule(Messages::value_type&& message)
  {
    auto const* header = message->header();
    if (header == nullptr) {
      throw std::logic_error("No valid header message found");
    }
    mScheduledMessages.emplace_back(std::move(message));
    if (mDispatchControl.dispatch != nullptr) {
      // send all scheduled messages if there is no trigger callback or its result is true
      if (mDispatchControl.trigger == nullptr || mDispatchControl.trigger(*header)) {
        std::vector<fair::mq::Parts> outputsPerChannel;
        outputsPerChannel.resize(mProxy.getNumOutputChannels());
        for (auto& message : mScheduledMessages) {
          fair::mq::Parts parts = message->finalize();
          assert(message->empty());
          assert(parts.Size() == 2);
          for (auto& part : parts) {
            outputsPerChannel[mProxy.getOutputChannelIndex(message->route()).value].AddPart(std::move(part));
          }
        }
        for (int ci = 0; ci < mProxy.getNumOutputChannels(); ++ci) {
          auto& parts = outputsPerChannel[ci];
          if (parts.Size() == 0) {
            continue;
          }
          mDispatchControl.dispatch(std::move(parts), ChannelIndex{ci}, DefaultChannelIndex);
        }
        mDidDispatch = mScheduledMessages.empty() == false;
        mScheduledMessages.clear();
      }
    }
  }

  Messages getMessagesForSending()
  {
    // before starting iteration, message lists are merged
    for (auto& message : mScheduledMessages) {
      mMessages.emplace_back(std::move(message));
    }
    mScheduledMessages.clear();
    return std::move(mMessages);
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
    for (auto& m : mMessages) {
      assert(m->empty());
    }
    mMessages.clear();
  }

  FairMQDeviceProxy& proxy()
  {
    return mProxy;
  }

  // Add a message to cache and returns a unique identifier for
  // such cached message.
  int64_t addToCache(std::unique_ptr<fair::mq::Message>& message);
  // Clone a message from cache so that it can be added to the context
  [[nodiscard]] std::unique_ptr<fair::mq::Message> cloneFromCache(int64_t id) const;
  // Prune a message from cache
  void pruneFromCache(int64_t id);

  /// call the proxy to create a message of the specified size
  /// we don't implement in the header to avoid including the fair::mq::Device header here
  /// that's why the different versions need to be implemented as individual functions
  // FIXME: can that be const?
  fair::mq::MessagePtr createMessage(RouteIndex routeIndex, int index, size_t size);
  fair::mq::MessagePtr createMessage(RouteIndex routeIndex, int index, void* data, size_t size, fair::mq::FreeFn* ffn, void* hint);

  /// return the headers of the 1st (from the end) matching message checking first in mMessages then in mScheduledMessages
  o2::header::DataHeader* findMessageHeader(const Output& spec);
  o2::header::Stack* findMessageHeaderStack(const Output& spec);
  int countDeviceOutputs(bool excludeDPLOrigin = false);
  void fakeDispatch() { mDidDispatch = true; }
  o2::framework::DataProcessingHeader* findMessageDataProcessingHeader(const Output& spec);
  std::pair<o2::header::DataHeader*, o2::framework::DataProcessingHeader*> findMessageHeaders(const Output& spec);

 private:
  FairMQDeviceProxy& mProxy;
  Messages mMessages;
  Messages mScheduledMessages;
  bool mDidDispatch = false;
  DispatchControl mDispatchControl;
  /// Cached messages, in case we want to reuse them.
  std::unordered_map<int64_t, std::unique_ptr<fair::mq::Message>> mMessageCache;
};
} // namespace o2::framework
#endif // O2_FRAMEWORK_MESSAGECONTEXT_H_
