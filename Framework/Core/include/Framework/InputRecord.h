// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_INPUTRECORD_H
#define FRAMEWORK_INPUTRECORD_H

#include "Framework/DataRef.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRoute.h"
#include "Framework/TypeTraits.h"
#include "Framework/InputSpan.h"
#include "Framework/TableConsumer.h"

#include "CommonUtils/BoostSerializer.h"

#include <gsl/gsl>

#include <iterator>
#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include <exception>
#include <memory>
#include <type_traits>

class FairMQMessage;

namespace o2
{
namespace framework
{

struct InputSpec;

/// @class InputRecord
/// @brief The input API of the Data Processing Layer
/// This class holds the inputs which  are being processed by the system while
/// they are  being processed.  The user can  get an instance  for it  via the
/// ProcessingContext and can use it to retrieve the inputs, either by name or
/// by index.  A few utility  methods are  provided to automatically  cast the
/// inputs to  known types. The user is also allowed to  override the `get`
/// template and provide his own serialization mechanism.
///
/// The @ref get<T>(binding) method is implemeted for the following types:
/// - (a) const char*
/// - (b) std::string
/// - (c) messageable type T
/// - (d) pointer type T* for types with ROOT dictionary or messageable types
/// - (e) std container of type T with ROOT dictionary
/// - (f) DataRef holding header and payload information, this is also the default
///       get method without template parameter
///
/// The return type of get<T>(binding) is:
/// - (a) const char* to payload content
/// - (b) std::string copy of the payload
/// - (c) const ref for messageable types T, the payload content casted to the type
/// - (d) object with pointer-like behavior (unique_ptr) if T* specified
/// - (e) std::container object returned by std::move
/// - (f) DataRef object returned by copy
///
/// Iterator functionality is implemented to iterate over the list of DataRef objects,
/// including begin() and end() methods.
/// <pre>
///    for (auto const& ref : inputs) {
///      // do something with DataRef object ref
///    }
/// </pre>
class InputRecord
{
 public:
  InputRecord(std::vector<InputRoute> const& inputs,
              InputSpan&& span);

  /// A deleter type to be used with unique_ptr, which can be marked that
  /// it does not own the underlying resource and thus should not delete it.
  /// The resource ownership property controls the behavior and can only be
  /// set at construction of the deleter in the unique_ptr. Falls back to
  /// default_delete if not initialized to 'NotOwning'.
  /// Usage: unique_ptr<T, Deleter<T>> ptr(..., Deleter<T>(false))
  ///
  /// By contract, the underlying, not owned resource is supposed to be
  /// available during the lifetime of the object, which is the case in the
  /// InputRecord and DPL processing APIs. The Deleter can be extended
  /// to support a callback to call a resource management outside.
  template <typename T>
  class Deleter : public std::default_delete<T>
  {
   public:
    enum struct OwnershipProperty : short {
      Unknown = -1,
      NotOwning = 0, /// don't delete the underlying buffer
      Owning = 1     /// normal behavior, falling back to default deleter
    };

    using base = std::default_delete<T>;
    using self_type = Deleter<T>;
    // using pointer = typename base::pointer;

    constexpr Deleter() = default;
    constexpr Deleter(bool isOwning)
      : base::default_delete(), mProperty(isOwning ? OwnershipProperty::Owning : OwnershipProperty::NotOwning)
    {
    }

    // copy constructor is needed in the setup of unique_ptr
    // check that assignments happen only to uninitialized instances
    constexpr Deleter(const self_type& other)
    {
      if (mProperty == OwnershipProperty::Unknown) {
        mProperty = other.mProperty;
      } else if (mProperty != other.mProperty) {
        throw std::runtime_error("Attemp to change resource control");
      }
    }

    // copy constructor for the default delete which simply sets the
    // resource ownership control to 'Owning'
    constexpr Deleter(const base& other) { mProperty = OwnershipProperty::Owning; }

    // forbid assignment operator to prohibid changing the Deleter
    // resource control property once used in the unique_ptr
    self_type& operator=(const self_type&) = delete;

    void operator()(T* ptr) const
    {
      if (mProperty == OwnershipProperty::NotOwning)
        return;
      base::operator()(ptr);
    }

   private:
    OwnershipProperty mProperty = OwnershipProperty::Unknown;
  };

  int getPos(const char* name) const;
  int getPos(const std::string& name) const;

  DataRef getByPos(int pos) const
  {
    if (pos * 2 + 1 > mSpan.size() || pos < 0) {
      throw std::runtime_error("Unknown message requested at position " + std::to_string(pos));
    }
    if (pos > mInputsSchema.size()) {
      throw std::runtime_error("Unknown schema at position" + std::to_string(pos));
    }
    if (mSpan.get(pos * 2) != nullptr && mSpan.get(pos * 2 + 1) != nullptr) {
      return DataRef{&mInputsSchema[pos].matcher,
                     mSpan.get(pos * 2),
                     mSpan.get(pos * 2 + 1)};
    } else {
      return DataRef{&mInputsSchema[pos].matcher, nullptr, nullptr};
    }
  }

  /// Generic function to extract a messageable type by reference
  /// Cast content of payload bound by @a binding to known type.
  /// Will not be used for types needing extra serialization
  /// Note: is_messagable also checks that T is not a pointer
  /// @return const ref to specified type
  template <typename T>
  typename std::enable_if<is_messageable<T>::value && std::is_same<T, DataRef>::value == false && framework::is_boost_serializable<T>::value == false, //
                          T>::type const&
    get(char const* binding) const
  {
    // we need to check the serialization type, the cast makes only sense for
    // unserialized objects
    // FIXME: add specialization for move assignable types
    using DataHeader = o2::header::DataHeader;

    auto ref = this->get(binding);
    auto header = o2::header::get<const DataHeader*>(ref.header);
    auto method = header->payloadSerializationMethod;
    if (method != o2::header::gSerializationMethodNone) {
      // FIXME: we could in principle support serialized content here as well if we
      // store all extracted objects internally and provide cleanup
      throw std::runtime_error("Can not extract a plain object from serialized message");
    }
    return *reinterpret_cast<T const*>(get<DataRef>(binding).payload);
  }

  /// Generic function to extract a messageable type by pointer
  /// @return unique_ptr to message content with custom deleter
  template <class PtrT>
  typename std::enable_if<                                                                     //
    std::is_pointer<PtrT>::value == true && std::is_same<PtrT, const char*>::value == false && //
      is_messageable<typename std::remove_pointer<PtrT>::type>::value == true &&               //
      has_root_dictionary<typename std::remove_pointer<PtrT>::type>::value == false,           //
    std::unique_ptr<typename std::remove_pointer<PtrT>::type const,                            //
                    Deleter<typename std::remove_pointer<PtrT>::type const>>>::type            //
    get(char const* binding) const
  {
    using T = typename std::remove_pointer<PtrT>::type;
    auto const& data = get<T>(binding);
    // return type with non-owning Deleter instance
    std::unique_ptr<T const, Deleter<T const>> result(&data, Deleter<T const>(false));
    return std::move(result);
  }

  /// substitution for const char*
  /// If we ask for a char const *, we simply point to the payload. Notice this
  /// is meant for C-style strings. If you want to actually get hold of the buffer,
  /// use get<DataRef> (or simply get) as that will give you the size as well.
  /// FIXME: check that the string is null terminated.
  /// @return pointer to payload content
  template <typename T>
  typename std::enable_if<std::is_same<T, char const*>::value, T>::type
    get(char const* binding) const
  {
    return reinterpret_cast<char const*>(get<DataRef>(binding).payload);
  }

  /// substitution for span of messageable objects
  /// Note: there is no check for serialization type for the moment, which means that the method
  /// can be used to get the raw buffer by simply querying gsl::span<unsigned char>.
  /// FIXME: there will be std::span in C++20
  template <typename T>
  typename std::enable_if<std::is_same<T, gsl::span<typename T::value_type>>::value == true,
                          gsl::span<typename T::value_type const>>::type
    get(char const* binding) const
  {
    auto&& ref = get<DataRef>(binding);
    auto header = header::get<const header::DataHeader*>(ref.header);
    assert(header);
    using ValueT = typename T::value_type;
    if (header->payloadSize % sizeof(ValueT)) {
      throw std::runtime_error("Inconsistent type and payload size at " + std::string(binding) +
                               ": type size " + std::to_string(sizeof(ValueT)) +
                               "  payload size " + std::to_string(header->payloadSize));
    }
    return gsl::span<ValueT const>(reinterpret_cast<ValueT const*>(ref.payload), header->payloadSize / sizeof(ValueT));
  }

  /// substitution for std::string
  /// If we ask for a string, we need to duplicate it because we do not want
  /// the buffer to be deleted when it goes out of scope.
  /// FIXME: check that the string is null terminated.
  /// @return std::string object
  template <typename T>
  typename std::enable_if<std::is_same<T, std::string>::value, T>::type
    get(char const* binding) const
  {
    auto&& ref = get<DataRef>(binding);
    auto header = header::get<const header::DataHeader*>(ref.header);
    assert(header);
    return std::move(std::string(ref.payload, header->payloadSize));
  }

  /// substitution for TableConsumer
  /// For the moment this is dummy, as it requires proper support to
  /// create the RDataSource from the arrow buffer.
  template <typename T>
  typename std::enable_if<std::is_same<T, TableConsumer>::value, std::unique_ptr<TableConsumer>>::type
    get(char const* binding) const
  {

    auto&& ref = get<DataRef>(binding);
    auto header = header::get<const header::DataHeader*>(ref.header);
    assert(header);
    auto data = reinterpret_cast<uint8_t const*>(ref.payload);
    return std::move(std::make_unique<TableConsumer>(data, header->payloadSize));
  }

  /// substitution for boost-serialized entities
  /// We have to deserialize the ostringstream.
  /// FIXME: check that the string is null terminated.
  /// @return deserialized copy of payload
  template <typename T>
  typename std::enable_if<framework::is_boost_serializable<T>::value == true //
                            && std::is_same<T, std::string>::value == false  //
                            && has_root_dictionary<T>::value == false,
                          T>::type
    get(char const* binding) const
  {
    auto&& ref = get<DataRef>(binding);
    auto header = header::get<const header::DataHeader*>(ref.header);
    assert(header);
    auto str = std::string(ref.payload, header->payloadSize);
    assert(header->payloadSize == sizeof(T));
    auto desData = o2::utils::BoostDeserialize<T>(str);
    return std::move(desData);
  }

  template <typename T, typename WT = typename T::wrapped_type>
  typename std::enable_if<is_specialization<T, BoostSerialized>::value == true, WT>::type
    get(char const* binding) const
  {
    auto&& ref = get<DataRef>(binding);
    auto header = header::get<const header::DataHeader*>(ref.header);
    assert(header);
    auto str = std::string(ref.payload, header->payloadSize);
    auto desData = o2::utils::BoostDeserialize<WT>(str);
    return std::move(desData);
  }

  template <typename T>
  T get_boost(char const* binding) const
  {
    auto&& ref = get<DataRef>(binding);
    auto header = header::get<const header::DataHeader*>(ref.header);
    assert(header);
    auto str = std::string(ref.payload, header->payloadSize);
    auto desData = o2::utils::BoostDeserialize<T>(str);
    return std::move(desData);
  }

  /// substitution for DataRef
  /// DataRef is special. Since there is no point in storing one in a payload,
  /// what it actually does is to return the DataRef used to hold the
  /// (header, payload) pair.
  /// @return DataRef object
  template <typename T = DataRef>
  typename std::enable_if<std::is_same<T, DataRef>::value, T>::type
    get(const char* binding) const
  {
    try {
      auto pos = getPos(binding);
      if (pos < 0) {
        throw std::invalid_argument("no matching route found for " + std::string(binding));
      }
      return getByPos(pos);
    } catch (const std::exception& e) {
      throw std::runtime_error("Unknown argument requested " + std::string(binding) +
                               " - " + e.what());
    }
  }

  /// substitution for DataRef
  /// DataRef is special. Since there is no point in storing one in a payload,
  /// what it actually does is to return the DataRef used to hold the
  /// (header, payload) pair.
  /// @return DataRef object
  template <class T = DataRef>
  typename std::enable_if<std::is_same<T, DataRef>::value, T>::type
    get(std::string const& binding) const
  {
    try {
      auto pos = getPos(binding);
      if (pos < 0) {
        throw std::invalid_argument("no matching route found for " + binding);
      }
      return getByPos(pos);
    } catch (const std::exception& e) {
      throw std::runtime_error("Unknown argument requested " + std::string(binding) +
                               " - " + e.what());
    }
  }

  /// substitution for pointer to non-messageable objects with ROOT dictionary
  /// Template parameter is a pointer
  /// This supports the common case of retrieving a root object and getting pointer.
  /// Notice that this will return a copy of the actual contents of the buffer, because
  /// the buffer is actually serialised, for this reason we return a unique_ptr<T>.
  /// FIXME: does it make more sense to keep ownership of all the deserialised
  /// objects in a single place so that we can avoid duplicate deserializations?
  /// @return unique_ptr to deserialized content
  template <class PtrT>
  typename std::enable_if<                                                            //
    std::is_pointer<PtrT>::value == true &&                                           //
      has_root_dictionary<typename std::remove_pointer<PtrT>::type>::value == true && //
      is_messageable<typename std::remove_pointer<PtrT>::type>::value == false,       //
    std::unique_ptr<typename std::remove_pointer<PtrT>::type const>>::type            //
    get(char const* binding) const
  {
    using T = typename std::remove_pointer<PtrT>::type;
    auto ref = this->get(binding);
    return std::move(DataRefUtils::as<T>(ref));
  }

  /// substitution for container of non-messageable objects with ROOT dictionary
  /// Notice that this will return a copy of the actual contents of the buffer, because
  /// the buffer is actually serialised. The extracted container is swaped to local,
  /// container, C++11 and beyond will implicitly apply return value optimization.
  /// @return std container object
  template <class T>
  typename std::enable_if<is_container<T>::value == true &&          //
                            has_root_dictionary<T>::value == true && //
                            is_messageable<T>::value == false,       //
                          T>::type                                   //
    get(char const* binding) const
  {
    using NonConstT = typename std::remove_const<T>::type;
    auto ref = this->get(binding);
    // we expect the unique_ptr to hold an object, exception should have been thrown
    // otherwise
    auto object = DataRefUtils::as<NonConstT>(ref);
    // need to swap the content of the deserialized container to a local variable to force return
    // value optimization
    NonConstT container;
    std::swap(container, *object);
    return container;
  }

  /// substitution for pointer to messageable objects with ROOT dictionary
  /// the operation depends on the transmitted serialization method
  /// @return unique_ptr to deserialized content
  template <typename PtrT>
  typename std::enable_if<std::is_pointer<PtrT>::value == true &&                                           //
                            has_root_dictionary<typename std::remove_pointer<PtrT>::type>::value == true && //
                            is_messageable<typename std::remove_pointer<PtrT>::type>::value == true,        //
                          std::unique_ptr<typename std::remove_pointer<PtrT>::type const,
                                          Deleter<typename std::remove_pointer<PtrT>::type const>>>::type
    get(char const* binding) const
  {
    using DataHeader = o2::header::DataHeader;
    using T = typename std::remove_pointer<PtrT>::type;

    auto ref = this->get(binding);
    auto header = o2::header::get<const DataHeader*>(ref.header);
    auto method = header->payloadSerializationMethod;
    if (method == o2::header::gSerializationMethodNone) {
      auto const* ptr = reinterpret_cast<T const*>(ref.payload);
      // return type with non-owning Deleter instance
      std::unique_ptr<T const, Deleter<T const>> result(ptr, Deleter<T const>(false));
      return std::move(result);
    } else if (method == o2::header::gSerializationMethodROOT) {
      // explicitely specify serialization method to ROOT-serialized because type T
      // is messageable and a different method would be deduced in DataRefUtils
      // return type with owning Deleter instance, forwarding to default_deleter
      std::unique_ptr<T const, Deleter<T const>> result(DataRefUtils::as<ROOTSerialized<T>>(ref).release());
      return std::move(result);
    } else {
      throw std::runtime_error("Attempt to extract object from message with unsupported serialization type");
    }
  }

  // substitution for non-messageable objects for which serialization method can not
  // be derived by type, the operation depends on the transmitted serialization method
  // FIXME: some of the substitutions can for sure be combined when the return types
  // will be unified in a later refactoring
  // FIXME: request a pointer where you get a pointer
  template <typename T>
  typename std::enable_if_t<is_messageable<T>::value == false                         //
                              && std::is_pointer<T>::value == false                   //
                              && std::is_same<T, DataRef>::value == false             //
                              && std::is_same<T, std::string>::value == false         //
                              && has_root_dictionary<T>::value == false               //
                              && framework::is_boost_serializable<T>::value == false, //
                            std::unique_ptr<T const, Deleter<T const>>>::type
    get(char const* binding) const
  {
    using DataHeader = o2::header::DataHeader;

    auto ref = this->get(binding);
    auto header = o2::header::get<const DataHeader*>(ref.header);
    auto method = header->payloadSerializationMethod;
    if (method == o2::header::gSerializationMethodNone) {
      throw std::runtime_error(
        "Type mismatch: attempt to extract a non-messagable object from message with unserialized data");
    } else if (method == o2::header::gSerializationMethodROOT) {
      // explicitely specify serialization method to ROOT-serialized because type T
      // is messageable and a different method would be deduced in DataRefUtils
      // return type with owning Deleter instance, forwarding to default_deleter
      std::unique_ptr<T const, Deleter<T const>> result(DataRefUtils::as<ROOTSerialized<T>>(ref).release());
      return std::move(result);
    } else {
      throw std::runtime_error("Attempt to extract object from message with unsupported serialization type");
    }
  }

  /// Helper method to be used to check if a given part of the InputRecord is present.
  bool isValid(std::string const& s)
  {
    return isValid(s.c_str());
  }

  /// Helper method to be used to check if a given part of the InputRecord is present.
  bool isValid(char const* s);
  bool isValid(int pos);

  size_t size() const
  {
    return mSpan.size() / 2;
  }

  template <typename T>
  using IteratorBase = std::iterator<std::forward_iterator_tag, T>;

  template <typename ParentT, typename T>
  class Iterator : public IteratorBase<T>
  {
   public:
    using ParentType = ParentT;
    using SelfType = Iterator;
    using value_type = typename IteratorBase<T>::value_type;
    using reference = typename IteratorBase<T>::reference;
    using pointer = typename IteratorBase<T>::pointer;
    using ElementType = typename std::remove_const<value_type>::type;

    Iterator() = delete;

    Iterator(ParentType const* parent, size_t position = 0, size_t size = 0)
      : mParent(parent), mPosition(position), mSize(size > position ? size : position), mElement{nullptr, nullptr, nullptr}
    {
      if (mPosition < mSize) {
        mElement = mParent->getByPos(mPosition);
      }
    }

    ~Iterator() = default;

    // prefix increment
    SelfType& operator++()
    {
      if (mPosition < mSize && ++mPosition < mSize) {
        mElement = mParent->getByPos(mPosition);
      }
      return *this;
    }
    // postfix increment
    SelfType operator++(int /*unused*/)
    {
      SelfType copy(*this);
      operator++();
      return copy;
    }
    // return reference
    reference operator*()
    {
      return mElement;
    }
    // comparison
    bool operator==(const SelfType& rh)
    {
      return mPosition == rh.mPosition;
    }
    // comparison
    bool operator!=(const SelfType& rh)
    {
      return mPosition != rh.mPosition;
    }

   private:
    size_t mPosition;
    size_t mSize;
    ParentType const* mParent;
    ElementType mElement;
  };

  using iterator = Iterator<InputRecord, DataRef>;
  using const_iterator = Iterator<InputRecord, const DataRef>;

  const_iterator begin() const
  {
    return const_iterator(this, 0, size());
  }

  const_iterator end() const
  {
    return const_iterator(this, size());
  }

 private:
  std::vector<InputRoute> const& mInputsSchema;
  InputSpan mSpan;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_INPUTREGISTRY_H
