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
#include "Framework/TMessageSerializer.h"
#include "MemoryResources/MemoryResources.h"
#include "Headers/DataHeader.h"

#include <fairmq/FairMQMessage.h>
#include <fairmq/FairMQParts.h>

#include <vector>
#include <cassert>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <functional>

class FairMQDevice;

namespace o2
{
namespace framework
{

class MessageContext {
 public:
  using DispatchCallback = std::function<void(FairMQParts&& message, std::string const&, int)>;
  // so far we are only using one instance per named channel
  static constexpr int DefaultChannelIndex = 0;

  MessageContext(FairMQDeviceProxy proxy)
    : mProxy{ proxy }
  {
  }

  MessageContext(FairMQDeviceProxy proxy, DispatchCallback&& dispatcher)
    : mProxy{ proxy }, mDispatchCallback{ dispatcher }
  {
  }

  void init(DispatchCallback&& dispatcher)
  {
    mDispatchCallback = dispatcher;
  }

  // this is the virtual interface for context objects
  class ContextObject
  {
   public:
    ContextObject() = default;
    ContextObject(FairMQMessagePtr&& headerMsg, FairMQMessagePtr&& payloadMsg, const std::string& bindingChannel)
      : mParts{}, mChannel{ bindingChannel }
    {
      mParts.AddPart(std::move(headerMsg));
      mParts.AddPart(std::move(payloadMsg));
    }
    ContextObject(FairMQMessagePtr&& headerMsg, const std::string& bindingChannel)
      : mParts{}, mChannel{ bindingChannel }
    {
      mParts.AddPart(std::move(headerMsg));
    }
    virtual ~ContextObject() = default;

    /// @brief Finalize the object and return the parts by move
    /// This is the default method and can be overloaded by other implmentations to carry out other
    /// tasks before returning the parts objects
    virtual FairMQParts finalize()
    {
      FairMQParts parts = std::move(mParts);
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
    const std::string& channel() const
    {
      return mChannel;
    }

    bool empty() const
    {
      return mParts.Size() == 0;
    }

   protected:
    FairMQParts mParts;
    std::string mChannel;
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
      assert(mParts.Size() == 2);
      return mParts[1].GetData();
    }
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
    static_assert(std::is_same<typename T::allocator_type, o2::pmr::polymorphic_allocator<value_type>>::value, "container must have polymorphic allocator");
    /// default contructor forbidden, object always has to control message instances
    ContainerRefObject() = delete;
    /// constructor taking header message by move and creating the paypload message
    template <typename ContextType, typename... Args>
    ContainerRefObject(ContextType* context, FairMQMessagePtr&& headerMsg, const std::string& bindingChannel, int index, Args&&... args)
      : ContextObject(std::forward<FairMQMessagePtr>(headerMsg), bindingChannel),
        // the transport factory
        mFactory{ context->proxy().getTransport(bindingChannel, index) },
        // the memory resource takes ownership of the message
        mResource{ mFactory ? mFactory->GetMemoryResource() : nullptr },
        // create the vector with apropriate underlying memory resource for the message
        mData{ std::forward<Args>(args)..., pmr::polymorphic_allocator<value_type>(mResource) }
    {
      // FIXME: drop this repeated check and make sure at initial setup of devices that everything is fine
      // introduce error policy
      if (mFactory == nullptr) {
        throw std::runtime_error(std::string("failed to get transport factory for channel ") + bindingChannel);
      }
      if (mResource == nullptr) {
        throw std::runtime_error(std::string("no memory resource for channel ") + bindingChannel);
      }
    }
    ~ContainerRefObject() override = default;

    /// @brief Finalize object and return parts by move
    /// This retrieves the actual message from the vector object and moves it to the parts
    FairMQParts finalize() final
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
    FairMQTransportFactory* mFactory = nullptr;     /// pointer to transport factory
    pmr::FairMQMemoryResource* mResource = nullptr; /// message resource
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
    SpanObject(ContextType* context, FairMQMessagePtr&& headerMsg, const std::string& bindingChannel, int index, size_t nElements)
    {
      // create the span object for the memory of the payload message
      // TODO: we probably also want to check consistency of the header message, i.e. payloadSize member
      auto payloadMsg = context->createMessage(bindingChannel, index, nElements * sizeof(T));
      mValue = value_type(reinterpret_cast<T*>(payloadMsg->GetData()), nElements);
      mParts.AddPart(std::move(headerMsg));
      mParts.AddPart(std::move(payloadMsg));
      mChannel = bindingChannel;
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

  /// RootSerializedObject keeps ownership to an object which can be Root-serialized
  /// TODO: this should maybe be a separate header file to avoid including TMessageSerializer
  /// in this header file, but we can always change this without affecting to much code.
  template <typename T>
  class RootSerializedObject : public ContextObject
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
    RootSerializedObject(ContextType* context, FairMQMessagePtr&& headerMsg, const std::string& bindingChannel, Args&&... args)
      : ContextObject(std::forward<FairMQMessagePtr>(headerMsg), bindingChannel)
    {
      mObject = std::make_unique<value_type>(std::forward<Args>(args)...);
      mPayloadMsg = context->proxy().createMessage();
    }
    ~RootSerializedObject() override = default;

    /// @brief Finalize object and return parts by move
    /// This retrieves the actual message from the vector object and moves it to the parts
    FairMQParts finalize() final
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
    FairMQMessagePtr mPayloadMsg;
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
        throw std::runtime_error("No context available to schedule the context object");
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
  /// of the context if initialized. It is added to the list of messages otherwise.
  void schedule(Messages::value_type&& message)
  {
    if (mDispatchCallback) {
      mDispatchCallback(std::move(message->finalize()), message->channel(), DefaultChannelIndex);
      return;
    }
    mMessages.push_back(std::move(message));
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
      assert(m->empty());
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
  DispatchCallback mDispatchCallback;
};
} // namespace framework
} // namespace o2
#endif // FRAMEWORK_MESSAGECONTEXT_H
