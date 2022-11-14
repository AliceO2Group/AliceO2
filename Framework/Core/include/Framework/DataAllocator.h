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
#ifndef FRAMEWORK_DATAALLOCATOR_H
#define FRAMEWORK_DATAALLOCATOR_H

#include "Framework/MessageContext.h"
#include "Framework/StringContext.h"
#include "Framework/RawBufferContext.h"
#include "Framework/Output.h"
#include "Framework/OutputRef.h"
#include "Framework/OutputRoute.h"
#include "Framework/DataChunk.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/TimingInfo.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/TypeTraits.h"
#include "Framework/Traits.h"
#include "Framework/SerializationMethods.h"
#include "Framework/CheckTypes.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/RuntimeError.h"
#include "Framework/RouteState.h"

#include "Headers/DataHeader.h"
#include <TClass.h>
#include <gsl/span>

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <type_traits>
#include <utility>
#include <cstddef>

// Do not change this for a full inclusion of fair::mq::Device.
#include <fairmq/FwdDecls.h>

namespace arrow
{
class Schema;
class Table;

namespace ipc
{
class RecordBatchWriter;
} // namespace ipc
} // namespace arrow

namespace o2::framework
{
struct ServiceRegistry;

#define ERROR_STRING                                          \
  "data type T not supported by API, "                        \
  "\n specializations available for"                          \
  "\n - trivially copyable, non-polymorphic structures"       \
  "\n - arrays of those"                                      \
  "\n - TObject with additional constructor arguments"        \
  "\n - Classes and structs with boost serialization support" \
  "\n - std containers of those"
/// This allocator is responsible to make sure that the messages created match
/// the provided spec and that depending on how many pipelined reader we
/// have, messages get created on the channel for the reader of the current
/// timeframe.
class DataAllocator
{
 public:
  constexpr static ServiceKind service_kind = ServiceKind::Stream;
  using AllowedOutputRoutes = std::vector<OutputRoute>;
  using DataHeader = o2::header::DataHeader;
  using DataOrigin = o2::header::DataOrigin;
  using DataDescription = o2::header::DataDescription;
  using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;
  template <typename T>
  struct UninitializedVector {
    static_assert(std::is_fundamental<T>::value, "UninitializedVector only allowed for fundamental types");
    using value_type = T;
  };

  DataAllocator(ServiceRegistryRef ref);

  DataChunk& newChunk(const Output&, size_t);

  inline DataChunk& newChunk(OutputRef&& ref, size_t size) { return newChunk(getOutputByBind(std::move(ref)), size); }

  void adoptChunk(const Output&, char*, size_t, fair::mq::FreeFn*, void*);

  /// Generic helper to create an object which is owned by the framework and
  /// returned as a reference to the own object.
  /// Note: decltype(auto) will deduce the return type from the expression and it
  /// will be lvalue reference for the framework-owned objects. Instances of local
  /// variables like shared_ptr will be returned by value/move/return value optimization.
  /// Objects created this way will be sent to the channel specified by @spec
  template <typename T, typename... Args>
  decltype(auto) make(const Output& spec, Args... args)
  {
    if constexpr (is_specialization_v<T, UninitializedVector>) {
      // plain buffer as polymorphic spectator std::vector, which does not run constructors / destructors
      using ValueType = typename T::value_type;
      auto& timingInfo = mRegistry.get<TimingInfo>();
      auto routeIndex = matchDataHeader(spec, timingInfo.timeslice);
      auto& context = mRegistry.get<MessageContext>();

      // Note: initial payload size is 0 and will be set by the context before sending
      fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodNone, 0);
      return context.add<MessageContext::VectorObject<ValueType, MessageContext::ContainerRefObject<std::vector<ValueType, o2::pmr::NoConstructAllocator<ValueType>>>>>(
                      std::move(headerMessage), routeIndex, 0, std::forward<Args>(args)...)
        .get();
    } else if constexpr (is_specialization_v<T, std::vector> && has_messageable_value_type<T>::value) {
      // this catches all std::vector objects with messageable value type before checking if is also
      // has a root dictionary, so non-serialized transmission is preferred
      using ValueType = typename T::value_type;
      auto& timingInfo = mRegistry.get<TimingInfo>();
      auto routeIndex = matchDataHeader(spec, timingInfo.timeslice);
      auto& context = mRegistry.get<MessageContext>();

      // Note: initial payload size is 0 and will be set by the context before sending
      fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodNone, 0);
      return context.add<MessageContext::VectorObject<ValueType>>(std::move(headerMessage), routeIndex, 0, std::forward<Args>(args)...).get();
    } else if constexpr (has_root_dictionary<T>::value == true && is_messageable<T>::value == false) {
      // Extended support for types implementing the Root ClassDef interface, both TObject
      // derived types and others
      auto& timingInfo = mRegistry.get<TimingInfo>();
      auto routeIndex = matchDataHeader(spec, timingInfo.timeslice);
      auto& context = mRegistry.get<MessageContext>();

      // Note: initial payload size is 0 and will be set by the context before sending
      fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodROOT, 0);
      return context.add<MessageContext::RootSerializedObject<T>>(std::move(headerMessage), routeIndex, std::forward<Args>(args)...).get();
    } else if constexpr (std::is_base_of_v<std::string, T>) {
      auto* s = new std::string(args...);
      adopt(spec, s);
      return *s;
    } else if constexpr (std::is_base_of_v<struct TableBuilder, T>) {
      TableBuilder* tb = nullptr;
      call_if_defined<struct TableBuilder>([&](auto* p) {
        tb = new std::decay_t<decltype(*p)>(args...);
        adopt(spec, tb);
      });
      return *tb;
    } else if constexpr (std::is_base_of_v<struct TreeToTable, T>) {
      void* t2tr = nullptr;
      call_if_defined<struct TreeToTable>([&](auto* p) {
        auto t2t = new std::decay_t<decltype(*p)>(args...);
        adopt(spec, t2t);
        t2tr = t2t;
      });
      return *reinterpret_cast<TreeToTable*>(t2tr);
    } else if constexpr (sizeof...(Args) == 0) {
      if constexpr (is_messageable<T>::value == true) {
        return *reinterpret_cast<T*>(newChunk(spec, sizeof(T)).data());
      } else {
        static_assert(always_static_assert_v<T>, ERROR_STRING);
      }
    } else if constexpr (sizeof...(Args) == 1) {
      using FirstArg = typename std::tuple_element<0, std::tuple<Args...>>::type;
      if constexpr (std::is_integral_v<FirstArg>) {
        if constexpr (is_messageable<T>::value == true) {
          auto [nElements] = std::make_tuple(args...);
          auto size = nElements * sizeof(T);
          auto& timingInfo = mRegistry.get<TimingInfo>();
          auto routeIndex = matchDataHeader(spec, timingInfo.timeslice);
          auto& context = mRegistry.get<MessageContext>();

          fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex, o2::header::gSerializationMethodNone, size);
          return context.add<MessageContext::SpanObject<T>>(std::move(headerMessage), routeIndex, 0, nElements).get();
        }
      } else if constexpr (std::is_same_v<FirstArg, std::shared_ptr<arrow::Schema>>) {
        if constexpr (std::is_base_of_v<arrow::ipc::RecordBatchWriter, T>) {
          auto [schema] = std::make_tuple(args...);
          std::shared_ptr<arrow::ipc::RecordBatchWriter> writer;
          create(spec, &writer, schema);
          return writer;
        }
      } else {
        static_assert(always_static_assert_v<T>, ERROR_STRING);
      }
    } else {
      static_assert(always_static_assert_v<T>, ERROR_STRING);
    }
  }

  template <typename T>
  T& make_boost(const Output& spec)
  {
    auto buff = new T{};
    adopt_boost(spec, buff);
    return *buff;
  }

  /// Adopt a string in the framework and serialize / send
  /// it to the consumers of @a spec once done.
  void
    adopt(const Output& spec, std::string*);

  /// Adopt a TableBuilder in the framework and serialise / send
  /// it as an Arrow table to all consumers of @a spec once done
  void
    adopt(const Output& spec, struct TableBuilder*);

  /// Adopt a Tree2Table in the framework and serialise / send
  /// it as an Arrow table to all consumers of @a spec once done
  void
    adopt(const Output& spec, struct TreeToTable*);

  /// Adopt an Arrow table and send it to all consumers of @a spec
  void
    adopt(const Output& spec, std::shared_ptr<class arrow::Table>);

  /// Send a snapshot of an object, depending on the object type it is serialized before.
  /// The method always takes a copy of the data, which will then be sent once the
  /// computation ends.
  /// Framework does not take ownership of the @a object. Changes to @a object
  /// after the call will not be sent.
  ///
  /// Supported types:
  /// - messageable types (trivially copyable, non-polymorphic
  /// - std::vector of messageable types
  /// - std::vector of pointers of messageable type
  /// - types with ROOT dictionary and implementing the ROOT ClassDef interface
  ///
  /// Note: for many use cases, especially for the messageable types, the `make` interface
  /// might be better suited as the objects are allocated directly in the underlying
  /// memory resource and the copy can be avoided.
  ///
  /// Note: messageable objects with ROOT dictionary are preferably sent unserialized.
  /// Use @a ROOTSerialized type wrapper to force ROOT serialization. Same applies to
  /// types which do not implement the ClassDef interface but have a dictionary.
  template <typename T>
  void snapshot(const Output& spec, T const& object)
  {
    auto& proxy = mRegistry.get<MessageContext>().proxy();
    fair::mq::MessagePtr payloadMessage;
    auto serializationType = o2::header::gSerializationMethodNone;
    RouteIndex routeIndex = matchDataHeader(spec, mRegistry.get<TimingInfo>().timeslice);
    if constexpr (is_messageable<T>::value == true) {
      // Serialize a snapshot of a trivially copyable, non-polymorphic object,
      payloadMessage = proxy.createOutputMessage(routeIndex, sizeof(T));
      memcpy(payloadMessage->GetData(), &object, sizeof(T));

      serializationType = o2::header::gSerializationMethodNone;
    } else if constexpr (is_specialization_v<T, std::vector> == true ||
                         (gsl::details::is_span<T>::value && has_messageable_value_type<T>::value)) {
      using ElementType = typename std::remove_pointer<typename T::value_type>::type;
      if constexpr (is_messageable<ElementType>::value) {
        // Serialize a snapshot of a std::vector of trivially copyable, non-polymorphic elements
        // Note: in most cases it is better to use the `make` function und work with the provided
        // reference object
        constexpr auto elementSizeInBytes = sizeof(ElementType);
        auto sizeInBytes = elementSizeInBytes * object.size();
        payloadMessage = proxy.createOutputMessage(routeIndex, sizeInBytes);

        if constexpr (std::is_pointer<typename T::value_type>::value == false) {
          // vector of elements
          if (object.data() && sizeInBytes) {
            memcpy(payloadMessage->GetData(), object.data(), sizeInBytes);
          }
        } else {
          // serialize vector of pointers to elements
          auto target = reinterpret_cast<unsigned char*>(payloadMessage->GetData());
          for (auto const& pointer : object) {
            memcpy(target, pointer, elementSizeInBytes);
            target += elementSizeInBytes;
          }
        }

        serializationType = o2::header::gSerializationMethodNone;
      } else if constexpr (has_root_dictionary<ElementType>::value) {
        return snapshot(spec, ROOTSerialized<T const>(object));
      } else {
        static_assert(always_static_assert_v<T>,
                      "value type of std::vector not supported by API, supported types:"
                      "\n - messageable tyeps (trivially copyable, non-polymorphic structures)"
                      "\n - pointers to those"
                      "\n - types with ROOT dictionary and implementing ROOT ClassDef interface");
      }
    } else if constexpr (has_root_dictionary<T>::value == true || is_specialization_v<T, ROOTSerialized> == true) {
      // Serialize a snapshot of an object with root dictionary
      payloadMessage = proxy.createOutputMessage(routeIndex);
      if constexpr (is_specialization_v<T, ROOTSerialized> == true) {
        // Explicitely ROOT serialize a snapshot of object.
        // An object wrapped into type `ROOTSerialized` is explicitely marked to be ROOT serialized
        // and is expected to have a ROOT dictionary. Availability can not be checked at compile time
        // for all cases.
        using WrappedType = typename T::wrapped_type;
        static_assert(std::is_same<typename T::hint_type, const char>::value ||
                        std::is_same<typename T::hint_type, TClass>::value ||
                        std::is_void<typename T::hint_type>::value,
                      "class hint must be of type TClass or const char");

        const TClass* cl = nullptr;
        if (object.getHint() == nullptr) {
          // get TClass info by wrapped type
          cl = TClass::GetClass(typeid(WrappedType));
        } else if (std::is_same<typename T::hint_type, TClass>::value) {
          // the class info has been passed directly
          cl = reinterpret_cast<const TClass*>(object.getHint());
        } else if (std::is_same<typename T::hint_type, const char>::value) {
          // get TClass info by optional name
          cl = TClass::GetClass(reinterpret_cast<const char*>(object.getHint()));
        }
        if (has_root_dictionary<WrappedType>::value == false && cl == nullptr) {
          if (std::is_same<typename T::hint_type, const char>::value) {
            throw runtime_error_f("ROOT serialization not supported, dictionary not found for type %s",
                                  reinterpret_cast<const char*>(object.getHint()));
          } else {
            throw runtime_error_f("ROOT serialization not supported, dictionary not found for type %s",
                                  typeid(WrappedType).name());
          }
        }
        TMessageSerializer().Serialize(*payloadMessage, &object(), cl);
      } else {
        TMessageSerializer().Serialize(*payloadMessage, &object, TClass::GetClass(typeid(T)));
      }
      serializationType = o2::header::gSerializationMethodROOT;
    } else {
      static_assert(always_static_assert_v<T>,
                    "data type T not supported by API, \n specializations available for"
                    "\n - trivially copyable, non-polymorphic structures"
                    "\n - std::vector of messageable structures or pointers to those"
                    "\n - types with ROOT dictionary and implementing ROOT ClassDef interface");
    }
    addPartToContext(std::move(payloadMessage), spec, serializationType);
  }

  /// Take a snapshot of a raw data array which can be either POD or may contain a serialized
  /// object (in such case the serialization method should be specified accordingly). Changes
  /// to the data after the call will not be sent.
  void snapshot(const Output& spec, const char* payload, size_t payloadSize,
                o2::header::SerializationMethod serializationMethod = o2::header::gSerializationMethodNone);

  /// make an object of type T and route to output specified by OutputRef
  /// The object is owned by the framework, returned reference can be used to fill the object.
  ///
  /// OutputRef descriptors are expected to be passed as rvalue, i.e. a temporary object in the
  /// function call
  template <typename T, typename... Args>
  decltype(auto) make(OutputRef&& ref, Args&&... args)
  {
    return make<T>(getOutputByBind(std::move(ref)), std::forward<Args>(args)...);
  }

  /// adopt an object of type T and route to output specified by OutputRef
  /// Framework takes ownership of the object
  ///
  /// OutputRef descriptors are expected to be passed as rvalue, i.e. a temporary object in the
  /// function call
  template <typename T>
  void adopt(OutputRef&& ref, T* obj)
  {
    return adopt(getOutputByBind(std::move(ref)), obj);
  }

  //get the memory resource associated with an output
  o2::pmr::FairMQMemoryResource* getMemoryResource(const Output& spec)
  {
    auto& timingInfo = mRegistry.get<TimingInfo>();
    auto& proxy = mRegistry.get<FairMQDeviceProxy>();
    RouteIndex routeIndex = matchDataHeader(spec, timingInfo.timeslice);
    return *proxy.getOutputTransport(routeIndex);
  }

  //make a stl (pmr) vector
  template <typename T, typename... Args>
  o2::pmr::vector<T> makeVector(const Output& spec, Args&&... args)
  {
    o2::pmr::FairMQMemoryResource* targetResource = getMemoryResource(spec);
    return o2::pmr::vector<T>{targetResource, std::forward<Args>(args)...};
  }

  struct CacheId {
    int64_t value;
  };

  template <typename ContainerT>
  CacheId adoptContainer(const Output& /*spec*/, ContainerT& /*container*/, bool /* cache  = false */, o2::header::SerializationMethod /* method = header::gSerializationMethodNone*/)
  {
    static_assert(always_static_assert_v<ContainerT>, "Container cannot be moved. Please make sure it is backed by a o2::pmr::FairMQMemoryResource");
    return {0};
  }

  /// Adopt a PMR container. Notice that the container must be moveable and
  /// eventually backed / by a o2::pmr::FairMQMemoryResource.
  /// @a spec where to send the message
  /// @a container the container whose resource needs to be sent
  /// @a cache: if true, the messages being sent are shallow copies of a cached
  ///           copy. The entry in the cache can be subsequently be sent using
  ///           the returned CacheId.
  /// @return a unique id of the adopted message which can be used to resend the
  ///         message or can be pruned via the DataAllocator::prune() method.
  template <typename ContainerT>
  CacheId adoptContainer(const Output& spec, ContainerT&& container, bool cache = false, o2::header::SerializationMethod method = header::gSerializationMethodNone);

  /// Adopt an already cached message, using an already provided CacheId.
  void adoptFromCache(Output const& spec, CacheId id, header::SerializationMethod method = header::gSerializationMethodNone);

  /// snapshot object and route to output specified by OutputRef
  /// Framework makes a (serialized) copy of object content.
  ///
  /// OutputRef descriptors are expected to be passed as rvalue, i.e. a temporary object in the
  /// function call
  template <typename... Args>
  auto snapshot(OutputRef&& ref, Args&&... args)
  {
    return snapshot(getOutputByBind(std::move(ref)), std::forward<Args>(args)...);
  }

  /// check if a certain output is allowed
  bool isAllowed(Output const& query);

  o2::header::DataHeader* findMessageHeader(const Output& spec)
  {
    return mRegistry.get<MessageContext>().findMessageHeader(spec);
  }

  o2::header::DataHeader* findMessageHeader(OutputRef&& ref)
  {
    return mRegistry.get<MessageContext>().findMessageHeader(getOutputByBind(std::move(ref)));
  }

  o2::header::Stack* findMessageHeaderStack(const Output& spec)
  {
    return mRegistry.get<MessageContext>().findMessageHeaderStack(spec);
  }

  o2::header::Stack* findMessageHeaderStack(OutputRef&& ref)
  {
    return mRegistry.get<MessageContext>().findMessageHeaderStack(getOutputByBind(std::move(ref)));
  }

  int countDeviceOutputs(bool excludeDPLOrigin = false)
  {
    return mRegistry.get<MessageContext>().countDeviceOutputs(excludeDPLOrigin) +
           mRegistry.get<RawBufferContext>().countDeviceOutputs(excludeDPLOrigin);
  }

 private:
  ServiceRegistryRef mRegistry;

  RouteIndex matchDataHeader(const Output& spec, size_t timeframeId);
  fair::mq::MessagePtr headerMessageFromOutput(Output const& spec,                                  //
                                               RouteIndex index,                                    //
                                               o2::header::SerializationMethod serializationMethod, //
                                               size_t payloadSize);                                 //

  Output getOutputByBind(OutputRef&& ref);
  void addPartToContext(fair::mq::MessagePtr&& payload,
                        const Output& spec,
                        o2::header::SerializationMethod serializationMethod);
};

template <typename ContainerT>
DataAllocator::CacheId DataAllocator::adoptContainer(const Output& spec, ContainerT&& container, bool cache, header::SerializationMethod method)
{
  // Find a matching channel, extract the message for it form the container
  // and put it in the queue to be sent at the end of the processing
  auto& timingInfo = mRegistry.get<TimingInfo>();
  auto routeIndex = matchDataHeader(spec, timingInfo.timeslice);

  auto& context = mRegistry.get<MessageContext>();
  auto* transport = mRegistry.get<FairMQDeviceProxy>().getOutputTransport(routeIndex);
  fair::mq::MessagePtr payloadMessage = o2::pmr::getMessage(std::forward<ContainerT>(container), *transport);
  fair::mq::MessagePtr headerMessage = headerMessageFromOutput(spec, routeIndex,         //
                                                               method,                   //
                                                               payloadMessage->GetSize() //
  );

  CacheId cacheId{0}; //
  if (cache) {
    // The message will be shallow cloned in the cache. Since the
    // clone is indistinguishable from the original, we can keep sending
    // the original.
    cacheId.value = context.addToCache(payloadMessage);
  }

  context.add<MessageContext::TrivialObject>(std::move(headerMessage), std::move(payloadMessage), routeIndex);
  return cacheId;
}

} // namespace o2::framework

#endif //FRAMEWORK_DATAALLOCATOR_H
