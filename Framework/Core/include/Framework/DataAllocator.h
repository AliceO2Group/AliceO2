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
#include "Framework/MessageContext.h"
#include "Framework/RootObjectContext.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/TypeTraits.h"

#include "fairmq/FairMQMessage.h"

#include <vector>
#include <map>
#include <string>
#include <utility>
#include <type_traits>
#include <gsl/span>

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
  using DataHeader = o2::header::DataHeader;
  using DataOrigin = o2::header::DataOrigin;
  using DataDescription = o2::header::DataDescription;
  using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;

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
  typename std::enable_if<std::is_pod<T>::value == true, gsl::span<T>>::type
  make(const OutputSpec &spec, size_t nElements) {
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
  make(const OutputSpec &spec, Args... args) {
    auto obj = new T(args...);
    adopt(spec, obj);
    return *obj;
  }

  /// catching unsupported type for case without additional arguments
  /// have to add three specializations because of the different role of
  /// the arguments and the different return types
  template <typename T>
  typename std::enable_if<
    std::is_base_of<TObject, T>::value == false &&
    std::is_pod<T>::value == false,
    T&>::type
  make(const OutputSpec &) {
    static_assert(std::is_pod<T>::value == true ||
                  std::is_base_of<TObject, T>::value == true,
                  "data type T not supported by API, \n specializations available for"
                  "\n - POD structures and arrays of those"
                  "\n - TObject with additional constructor arguments");
  }

  /// catching unsupported type for case of span of objects
  template <typename T>
  typename std::enable_if<
    std::is_base_of<TObject, T>::value == false &&
    std::is_pod<T>::value == false,
    gsl::span<T>>::type
  make(const OutputSpec &, size_t) {
    static_assert(std::is_pod<T>::value == true,
                  "data type T not supported by API, \n specializations available for"
                  "\n - POD structures and arrays of those"
                  "\n - TObject with additional constructor arguments");
  }

  /// catching unsupported type for case of at least two additional arguments
  template <typename T, typename U, typename V, typename... Args>
  typename std::enable_if<
    std::is_base_of<TObject, T>::value == false &&
    std::is_pod<T>::value == false,
    T&>::type
  make(const OutputSpec &, U, V, Args...) {
    static_assert(std::is_pod<T>::value == true ||
                  std::is_base_of<TObject, T>::value == true,
                  "data type T not supported by API, \n specializations available for"
                  "\n - POD structures and arrays of those"
                  "\n - TObject with additional constructor arguments");
  }

  /// Adopt a TObject in the framework and serialize / send
  /// it to the consumers of @a spec once done.
  void
  adopt(const OutputSpec &spec, TObject*obj);

  /// Serialize a snapshot of a TObject when called, which will then be
  /// sent once the computation ends.
  /// Framework does not take ownership of the @a object. Changes to @a object
  /// after the call will not be sent.
  template <typename T>
  typename std::enable_if<std::is_base_of<TObject, T>::value == true, void>::type
  snapshot(const OutputSpec &spec, T & object) {
    FairMQMessagePtr payloadMessage(mDevice->NewMessage());
    mDevice->Serialize<TMessageSerializer>(*payloadMessage, &object);

    addPartToContext(std::move(payloadMessage), spec, o2::header::gSerializationMethodROOT);
  }

  /// Serialize a snapshot of a POD type, which will then be sent
  /// once the computation ends.
  /// Framework does not take ownership of @param object. Changes to @param object
  /// after the call will not be sent.
  template <typename T>
  typename std::enable_if<std::is_pod<T>::value == true, void>::type
  snapshot(const OutputSpec &spec, T const &object) {
    FairMQMessagePtr payloadMessage(mDevice->NewMessage(sizeof(T)));
    memcpy(payloadMessage->GetData(), &object, sizeof(T));

    addPartToContext(std::move(payloadMessage), spec, o2::header::gSerializationMethodNone);
  }

  /// Serialize a snapshot of a std::vector of trivially copyable elements,
  /// which will then be sent once the computation ends.
  /// Framework does not take ownership of @param object. Changes to @param object
  /// after the call will not be sent.
  template <typename C>
  typename std::enable_if<
    is_specialization<C, std::vector>::value == true
    && std::is_pointer<typename C::value_type>::value == false
    && std::is_trivially_copyable<typename C::value_type>::value == true
    >::type
  snapshot(const OutputSpec &spec, C const &v) {
    auto sizeInBytes = sizeof(typename C::value_type) * v.size();
    FairMQMessagePtr payloadMessage(mDevice->NewMessage(sizeInBytes));

    typename C::value_type *tmp = const_cast<typename C::value_type*>(v.data());
    memcpy(payloadMessage->GetData(), reinterpret_cast<void*>(tmp), sizeInBytes);

    addPartToContext(std::move(payloadMessage), spec, o2::header::gSerializationMethodNone);
  }

  /// Serialize a snapshot of a std::vector of pointers to trivially copyable
  /// elements, which will then be sent once the computation ends.
  /// Framework does not take ownership of @param object. Changes to @param object
  /// after the call will not be sent.
  template <typename C>
  typename std::enable_if<
    is_specialization<C, std::vector>::value == true
    && std::is_pointer<typename C::value_type>::value == true
    && std::is_trivially_copyable<typename std::remove_pointer<typename C::value_type>::type>::value == true
    >::type
  snapshot(const OutputSpec &spec, C const &v) {
    using ElementType = typename std::remove_pointer<typename C::value_type>::type;
    constexpr auto elementSizeInBytes = sizeof(ElementType);
    auto sizeInBytes = elementSizeInBytes * v.size();
    FairMQMessagePtr payloadMessage(mDevice->NewMessage(sizeInBytes));

    auto target = reinterpret_cast<unsigned char*>(payloadMessage->GetData());
    for (auto const & pointer : v) {
      memcpy(target, pointer, elementSizeInBytes);
      target += elementSizeInBytes;
    }

    addPartToContext(std::move(payloadMessage), spec, o2::header::gSerializationMethodNone);
  }

  /// specialization to catch unsupported types and throw a detailed compiler error
  template <typename T>
  typename std::enable_if<
    std::is_base_of<TObject, T>::value == false &&
    std::is_pod<T>::value == false &&
    is_specialization<T, std::vector>::value == false
    >::type
  snapshot(const OutputSpec &spec, T const &) {
    static_assert(std::is_base_of<TObject, T>::value == true ||
                  std::is_pod<T>::value == true ||
                  is_specialization<T, std::vector>::value == true,
                  "data type T not supported by API, \n specializations available for"
                  "\n - POD structures"
                  "\n - TObject"
                  "\n - std::vector of POD or pointer to POD");
  }

  /// specialization to catch unsupported types, check value_type of std::vector
  /// and throw a detailed compiler error
  template <typename T>
  typename std::enable_if<
    is_specialization<T, std::vector>::value == true &&
    std::is_trivially_copyable<
      typename std::remove_pointer<typename T::value_type>::type
      >::value == false
    >::type
  snapshot(const OutputSpec &spec, T const &) {
    static_assert(std::is_trivially_copyable<
                  typename std::remove_pointer<typename T::value_type>::type
                  >::value == true,
                  "data type T not supported by API, \n specializations available for"
                  "\n - POD structures"
                  "\n - TObject"
                  "\n - std::vector of POD or pointer to POD");
  }

private:
  std::string matchDataHeader(const OutputSpec &spec, size_t timeframeId);
  FairMQMessagePtr headerMessageFromSpec(OutputSpec const &spec,
                                         std::string const &channel,
                                         o2::header::SerializationMethod serializationMethod);

  void addPartToContext(FairMQMessagePtr&& payload,
                        const OutputSpec &spec,
                        o2::header::SerializationMethod serializationMethod);

  FairMQDevice *mDevice;
  AllowedOutputsMap mAllowedOutputs;
  MessageContext *mContext;
  RootObjectContext *mRootContext;
};

}
}

#endif //FRAMEWORK_DATAALLOCATOR_H
