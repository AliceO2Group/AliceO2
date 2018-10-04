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

#include "Headers/DataHeader.h"
#include "Framework/ContextRegistry.h"
#include "Framework/MessageContext.h"
#include "Framework/RootObjectContext.h"
#include "Framework/StringContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/Output.h"
#include "Framework/OutputRef.h"
#include "Framework/OutputRoute.h"
#include "Framework/DataChunk.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Framework/TimingInfo.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/TypeTraits.h"
#include "Framework/SerializationMethods.h"
#include "Framework/TableBuilder.h"

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <type_traits>
#include <gsl/span>
#include <utility>

#include <TClass.h>

// Do not change this for a full inclusion of FairMQDevice.
class FairMQDevice;
class FairMQMessage;

namespace o2
{
namespace framework
{

class ContextRegistry;

/// This allocator is responsible to make sure that the messages created match
/// the provided spec and that depending on how many pipelined reader we
/// have, messages get created on the channel for the reader of the current
/// timeframe.
class DataAllocator
{
public:
  using AllowedOutputRoutes = std::vector<OutputRoute>;
  using DataHeader = o2::header::DataHeader;
  using DataOrigin = o2::header::DataOrigin;
  using DataDescription = o2::header::DataDescription;
  using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

  DataAllocator(TimingInfo* timingInfo,
                ContextRegistry* contextes,
                const AllowedOutputRoutes& routes);

  DataChunk newChunk(const Output&, size_t);

  inline DataChunk newChunk(OutputRef&& ref, size_t size) { return newChunk(getOutputByBind(std::move(ref)), size); }

  DataChunk adoptChunk(const Output&, char *, size_t, fairmq_free_fn*, void *);

  // In case no extra argument is provided and the passed type is trivially
  // copyable and non polymorphic, the most likely wanted behavior is to create
  // a message with that type, and so we do.
  template <typename T>
  typename std::enable_if<is_messageable<T>::value == true, T&>::type
  make(const Output& spec)
  {
    DataChunk chunk = newChunk(spec, sizeof(T));
    return *reinterpret_cast<T*>(chunk.data);
  }

  // In case an extra argument is provided, we consider this an array / 
  // collection elements of that type
  template <typename T>
  typename std::enable_if<is_messageable<T>::value == true, gsl::span<T>>::type
  make(const Output& spec, size_t nElements)
  {
    auto size = nElements*sizeof(T);
    DataChunk chunk = newChunk(spec, size);
    return gsl::span<T>(reinterpret_cast<T*>(chunk.data), nElements);
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
  make(const Output& spec, Args... args) {
    auto obj = new T(args...);
    adopt(spec, obj);
    return *obj;
  }

  /// Helper to create an std::string which will be owned by the framework
  /// and transmitted when the processing finishes.
  template <typename T, typename... Args>
  typename std::enable_if<std::is_base_of<std::string, T>::value == true, T&>::type
    make(const Output& spec, Args... args)
  {
    std::string* s = new std::string(args...);
    adopt(spec, s);
    return *s;
  }

  /// Helper to create a TableBuilder which will be owned by the framework
  /// FIXME: perfect forwarding?
  template <typename T, typename... Args>
  typename std::enable_if<std::is_base_of<TableBuilder, T>::value == true, T &>::type
    make(const Output& spec, Args... args)
  {
    TableBuilder *tb = new TableBuilder(args...);
    adopt(spec, tb);
    return *tb;
  }

  /// catching unsupported type for case without additional arguments
  /// have to add three specializations because of the different role of
  /// the arguments and the different return types
  template <typename T>
  typename std::enable_if<
    std::is_base_of<TObject, T>::value == false
    && std::is_base_of<TableBuilder, T>::value == false
    && is_messageable<T>::value == false
    && std::is_same<std::string, T>::value == false,
    T&>::type
    make(const Output&)
  {
    static_assert(sizeof(T) == -1,
                  "data type T not supported by API, \n specializations available for"
                  "\n - trivially copyable, non-polymorphic structures"
                  "\n - arrays of those"
                  "\n - TObject with additional constructor arguments");
  }

  /// catching unsupported type for case of span of objects
  template <typename T>
  typename std::enable_if<
    std::is_base_of<TObject, T>::value == false
    && is_messageable<T>::value == false
    && std::is_same<std::string, T>::value == false,
    gsl::span<T>>::type
    make(const Output&, size_t)
  {
    static_assert(is_messageable<T>::value == true,
                  "data type T not supported by API, \n specializations available for"
                  "\n - trivially copyable, non-polymorphic structures"
                  "\n - arrays of those"
                  "\n - TObject with additional constructor arguments");
  }

  /// catching unsupported type for case of at least two additional arguments
  template <typename T, typename U, typename V, typename... Args>
  typename std::enable_if<
    std::is_base_of<TObject, T>::value == false
    && is_messageable<T>::value == false
    && std::is_same<std::string, T>::value == false,
    T&>::type
    make(const Output&, U, V, Args...)
  {
    static_assert(is_messageable<T>::value == true || std::is_base_of<TObject, T>::value == true,
                  "data type T not supported by API, \n specializations available for"
                  "\n - trivially copyable, non-polymorphic structures"
                  "\n - arrays of those"
                  "\n - TObject with additional constructor arguments");
  }

  /// Adopt a TObject in the framework and serialize / send
  /// it to the consumers of @a spec once done.
  void
  adopt(const Output& spec, TObject*obj);

  /// Adopt a string in the framework and serialize / send
  /// it to the consumers of @a spec once done.
  void
    adopt(const Output& spec, std::string*);

  /// Adopt a TableBuilder in the framework and serialise / send
  /// it as an Arrow table to all consumers of @a spec once done
  void
    adopt(const Output& spec, TableBuilder *);

  /// Serialize a snapshot of an object with root dictionary when called,
  /// will then be sent once the computation ends.
  /// Framework does not take ownership of the @a object. Changes to @a object
  /// after the call will not be sent.
  /// Note: also messageable objects can have a dictionary, but serialization
  /// method can not be deduced automatically. Messageable objects are sent
  /// unserialized by default. Serialization method needs to be specified
  /// explicitely otherwise by using ROOTSerialized wrapper type.
  template <typename T>
  typename std::enable_if<has_root_dictionary<T>::value == true && is_messageable<T>::value == false, void>::type
  snapshot(const Output& spec, T& object)
  {
    auto proxy = mContextRegistry->get<RootObjectContext>()->proxy();
    FairMQMessagePtr payloadMessage(proxy.createMessage());
    auto* cl = TClass::GetClass(typeid(T));
    TMessageSerializer().Serialize(*payloadMessage, &object, cl);

    addPartToContext(std::move(payloadMessage), spec, o2::header::gSerializationMethodROOT);
  }

  /// Explicitely ROOT serialize a snapshot of @a object when called,
  /// will then be sent once the computation ends. The @a object is wrapped
  /// into type ROOTSerialized to explicitely mark this serialization method,
  /// and is expected to have a ROOT dictionary. Availability can not be checked
  /// at compile time for all cases.
  /// Framework does not take ownership of the @a object. Changes to @a object
  /// after the call will not be sent.
  template <typename W>
  typename std::enable_if<is_specialization<W, ROOTSerialized>::value == true, void>::type
  snapshot(const Output& spec, W wrapper)
  {
    using T = typename W::wrapped_type;
    static_assert(std::is_same<typename W::hint_type, const char>::value || //
                    std::is_same<typename W::hint_type, TClass>::value ||   //
                    std::is_void<typename W::hint_type>::value,             //
                  "class hint must be of type TClass or const char");

    auto proxy = mContextRegistry->get<RootObjectContext>()->proxy();
    FairMQMessagePtr payloadMessage(proxy.createMessage());
    const TClass* cl = nullptr;
    if (wrapper.getHint() == nullptr) {
      // get TClass info by wrapped type
      cl = TClass::GetClass(typeid(T));
    } else if (std::is_same<typename W::hint_type, TClass>::value) {
      // the class info has been passed directly
      cl = reinterpret_cast<const TClass*>(wrapper.getHint());
    } else if (std::is_same<typename W::hint_type, const char>::value) {
      // get TClass info by optional name
      cl = TClass::GetClass(reinterpret_cast<const char*>(wrapper.getHint()));
    }
    if (has_root_dictionary<T>::value == false && cl == nullptr) {
      std::string msg("ROOT serialization not supported, dictionary not found for type ");
      if (std::is_same<typename W::hint_type, const char>::value) {
        msg += reinterpret_cast<const char*>(wrapper.getHint());
      } else {
        msg += typeid(T).name();
      }
      throw std::runtime_error(msg);
    }
    TMessageSerializer().Serialize(*payloadMessage, &wrapper(), cl);
    addPartToContext(std::move(payloadMessage), spec, o2::header::gSerializationMethodROOT);
  }

  /// Serialize a snapshot of a trivially copyable, non-polymorphic @a object,
  /// referred to be 'messageable, will then be sent once the computation ends.
  /// Framework does not take ownership of @param object. Changes to @param object
  /// after the call will not be sent.
  /// Note: also messageable objects with ROOT dictionary are preferably sent
  /// unserialized. Use @a ROOTSerialized type wrapper to force ROOT serialization.
  template <typename T>
  typename std::enable_if<is_messageable<T>::value == true, void>::type
  snapshot(const Output& spec, T const& object)
  {
    auto proxy = mContextRegistry->get<MessageContext>()->proxy();
    FairMQMessagePtr payloadMessage(proxy.createMessage(sizeof(T)));
    memcpy(payloadMessage->GetData(), &object, sizeof(T));

    addPartToContext(std::move(payloadMessage), spec, o2::header::gSerializationMethodNone);
  }

  /// Serialize a snapshot of a std::vector of trivially copyable, non-polymorphic
  /// elements, which will then be sent once the computation ends.
  /// Framework does not take ownership of @param object. Changes to @param object
  /// after the call will not be sent.
  template <typename C>
  typename std::enable_if<is_specialization<C, std::vector>::value == true &&
                          std::is_pointer<typename C::value_type>::value == false &&
                          is_messageable<typename C::value_type>::value == true>::type
  snapshot(const Output& spec, C const& v)
  {
    auto proxy = mContextRegistry->get<MessageContext>()->proxy();
    auto sizeInBytes = sizeof(typename C::value_type) * v.size();
    FairMQMessagePtr payloadMessage(proxy.createMessage(sizeInBytes));

    typename C::value_type *tmp = const_cast<typename C::value_type*>(v.data());
    memcpy(payloadMessage->GetData(), reinterpret_cast<void*>(tmp), sizeInBytes);

    addPartToContext(std::move(payloadMessage), spec, o2::header::gSerializationMethodNone);
  }

  /// Serialize a snapshot of a std::vector of pointers to trivially copyable,
  /// non-polymorphic elements, which will then be sent once the computation ends.
  /// Framework does not take ownership of @param object. Changes to @param object
  /// after the call will not be sent.
  template <typename C>
  typename std::enable_if<
    is_specialization<C, std::vector>::value == true &&
    std::is_pointer<typename C::value_type>::value == true &&
    is_messageable<typename std::remove_pointer<typename C::value_type>::type>::value == true>::type
  snapshot(const Output& spec, C const& v)
  {
    using ElementType = typename std::remove_pointer<typename C::value_type>::type;
    constexpr auto elementSizeInBytes = sizeof(ElementType);
    auto sizeInBytes = elementSizeInBytes * v.size();
    auto proxy = mContextRegistry->get<MessageContext>()->proxy();
    FairMQMessagePtr payloadMessage(proxy.createMessage(sizeInBytes));

    auto target = reinterpret_cast<unsigned char*>(payloadMessage->GetData());
    for (auto const & pointer : v) {
      memcpy(target, pointer, elementSizeInBytes);
      target += elementSizeInBytes;
    }

    addPartToContext(std::move(payloadMessage), spec, o2::header::gSerializationMethodNone);
  }

  /// specialization to catch unsupported types and throw a detailed compiler error
  template <typename T>
  typename std::enable_if<has_root_dictionary<T>::value == false &&                //
                          is_specialization<T, ROOTSerialized>::value == false &&  //
                          is_messageable<T>::value == false &&                     //
                          std::is_pointer<T>::value == false &&                    //
                          is_specialization<T, std::vector>::value == false>::type //
    snapshot(const Output& spec, T const&)
  {
    static_assert(has_root_dictionary<T>::value == true ||
                  is_specialization<T, ROOTSerialized>::value == true ||
                  is_messageable<T>::value == true ||
                  is_specialization<T, std::vector>::value == true,
                  "data type T not supported by API, \n specializations available for"
                  "\n - trivially copyable, non-polymorphic structures"
                  "\n - std::vector of messageable structures or pointers to those"
                  "\n - object with dictionary by reference");
  }

  /// specialization to catch unsupported types, check value_type of std::vector
  /// and throw a detailed compiler error
  template <typename T>
  typename std::enable_if<
    is_specialization<T, std::vector>::value == true &&
    is_messageable<
      typename std::remove_pointer<typename T::value_type>::type
      >::value == false
    >::type
  snapshot(const Output& spec, T const&)
  {
    static_assert(is_messageable<typename std::remove_pointer<typename T::value_type>::type>::value == true,
                  "data type T not supported by API, \n specializations available for"
                  "\n - trivially copyable, non-polymorphic structures"
                  "\n - std::vector of messageable structures or pointers to those"
                  "\n - object with dictionary by reference");
  }

  /// specialization to catch the case where a pointer to an object has been
  /// accidentally given as parameter
  template <typename T>
  typename std::enable_if<std::is_pointer<T>::value>::type snapshot(const Output& spec, T const&)
  {
    static_assert(std::is_pointer<T>::value == false,
                  "pointer to data type not supported by API. Please pass object by reference");
  }

  /// make an object of type T and route to output specified by OutputRef
  /// The object is owned by the framework, returned reference can be used to fill the object.
  ///
  /// OutputRef descriptors are expected to be passed as rvalue, i.e. a temporary object in the
  /// function call
  template <typename T, typename... Args>
  auto make(OutputRef&& ref, Args&&... args)
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

 private:
  AllowedOutputRoutes mAllowedOutputRoutes;
  TimingInfo *mTimingInfo;
  ContextRegistry* mContextRegistry;

  std::string matchDataHeader(const Output &spec, size_t timeframeId);
  FairMQMessagePtr headerMessageFromOutput(Output const& spec,                                  //
                                           std::string const& channel,                          //
                                           o2::header::SerializationMethod serializationMethod, //
                                           size_t payloadSize);                                 //

  Output getOutputByBind(OutputRef&& ref);
  void addPartToContext(FairMQMessagePtr&& payload,
                        const Output &spec,
                        o2::header::SerializationMethod serializationMethod);

};

} // namespace framework
} // namespace o2

#endif //FRAMEWORK_DATAALLOCATOR_H
