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
#include "Framework/Traits.h"
#include "MemoryResources/MemoryResources.h"
#include "Headers/DataHeader.h"

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
/// This class holds the inputs which are valid for processing. The user can get an
/// instance for it via the ProcessingContext and can use it to retrieve the inputs,
/// either by name or by index. A few utility methods are provided to automatically
/// cast the (deserialized) inputs to  known types.
///
/// \par The @ref get<T>(binding) method is implemeted for the following types:
/// - (a) @ref DataRef holding header and payload information, this is also the default
///       get method without template parameter
/// - (b) std::string
/// - (c) const char*
///       this is meant for C-style strings which are 0 terminated, there is no length
///       information
/// - (d) @ref TableConsumer
/// - (e) boost serializable types
/// - (f) span over messageable type T
/// - (g) std::vector of messageable type or type with ROOT dictionary
/// - (h) messageable type T
/// - (i) pointer type T* for types with ROOT dictionary or messageable types
///
/// \par The return type of get<T>(binding) is:
/// - (a) @ref DataRef object
/// - (b) std::string copy of the payload
/// - (c) const char* to payload content
/// - (d) unique_ptr of TableConsumer
/// - (e) object by move
/// - (f) span object over original payload
/// - (g) vector by move
/// - (h) reference to object
/// - (i) object with pointer-like behavior (unique_ptr)
///
/// \par Examples
/// <pre>
///    auto& v1 = get<int>("input1");
///    auto v2 = get<vector<int>>("input2");
///    auto v3 = get<TList*>("input3");
///    auto v4 = get<vector<TParticle>>("input4");
/// </pre>
///
/// \par Validity of inputs
/// Not all input slots are always valid if a custom completion policy is chosen. Validity
/// can be checked using method @ref isValid.
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
  using DataHeader = o2::header::DataHeader;

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

  /// get object of the specified type from input
  /// The actual operation and cast dependd on the target data type and the serialization type of the
  /// incoming data. See @ref Inputrecord class description for supported types.
  /// @param binding   the input to extract the data from
  template <typename T = DataRef>
  decltype(auto) get(char const* binding) const
  {
    // implementation (a)
    if constexpr (std::is_same<T, DataRef>::value) {
      // DataRef is special. Since there is no point in storing one in a payload,
      // what it actually does is to return the DataRef used to hold the
      // (header, payload) pair.
      // returns DataRef object
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

      // implementation (b)
    } else if constexpr (std::is_same<T, std::string>::value) {
      // substitution for std::string
      // If we ask for a string, we need to duplicate it because we do not want
      // the buffer to be deleted when it goes out of scope. The string is built
      // from the data and its lengh, null-termination is not necessary.
      // return std::string object
      auto&& ref = get<DataRef>(binding);
      auto header = header::get<const header::DataHeader*>(ref.header);
      assert(header);
      return std::string(ref.payload, header->payloadSize);

      // implementation (c)
    } else if constexpr (std::is_same<T, char const*>::value) {
      // substitution for const char*
      // If we ask for a char const *, we simply point to the payload. Notice this
      // is meant for C-style strings which are expected to be null terminated.
      // If you want to actually get hold of the buffer, use gsl::span<char> as that will
      // give you the size as well.
      // return pointer to payload content
      return reinterpret_cast<char const*>(get<DataRef>(binding).payload);

      // implementation (d)
    } else if constexpr (std::is_same<T, TableConsumer>::value) {
      // substitution for TableConsumer
      // For the moment this is dummy, as it requires proper support to
      // create the RDataSource from the arrow buffer.
      auto&& ref = get<DataRef>(binding);
      auto header = header::get<const header::DataHeader*>(ref.header);
      assert(header);
      auto data = reinterpret_cast<uint8_t const*>(ref.payload);
      return std::make_unique<TableConsumer>(data, header->payloadSize);

      // implementation (e)
    } else if constexpr (framework::is_boost_serializable<T>::value || is_specialization<T, BoostSerialized>::value) {
      // substitution for boost-serialized entities
      // We have to deserialize the ostringstream.
      // FIXME: check that the string is null terminated.
      // @return deserialized copy of payload
      auto&& ref = get<DataRef>(binding);
      auto header = header::get<const header::DataHeader*>(ref.header);
      assert(header);
      auto str = std::string(ref.payload, header->payloadSize);
      assert(header->payloadSize == sizeof(T));
      if constexpr (is_specialization<T, BoostSerialized>::value) {
        return o2::utils::BoostDeserialize<typename T::wrapped_type>(str);
      } else {
        return o2::utils::BoostDeserialize<T>(str);
      }

      // implementation (f)
    } else if constexpr (is_span<T>::value) {
      // substitution for span of messageable objects
      // Note: there is no check for serialization type for the moment, which means that the method
      // can be used to get the raw buffer by simply querying gsl::span<unsigned char>.
      // FIXME: there will be std::span in C++20
      static_assert(has_messageable_value_type<T>::value, "span can only be created for messageable types");
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

      // implementation (g)
    } else if constexpr (is_container<T>::value) {
      // currently implemented only for vectors
      if constexpr (is_specialization<typename std::remove_const<T>::type, std::vector>::value) {
        auto&& ref = get<DataRef>(binding);
        auto header = o2::header::get<const DataHeader*>(ref.header);
        auto method = header->payloadSerializationMethod;
        if (method == o2::header::gSerializationMethodNone) {
          // TODO: construct a vector spectator
          // this is a quick solution now which makes a copy of the plain vector data
          auto* start = reinterpret_cast<typename T::value_type const*>(ref.payload);
          auto* end = start + header->payloadSize / sizeof(typename T::value_type);
          T result(start, end);
          return result;
        } else if (method == o2::header::gSerializationMethodROOT) {
          /// substitution for container of non-messageable objects with ROOT dictionary
          /// Notice that this will return a copy of the actual contents of the buffer, because
          /// the buffer is actually serialised. The extracted container is swaped to local,
          /// container, C++11 and beyond will implicitly apply return value optimization.
          /// @return std container object
          using NonConstT = typename std::remove_const<T>::type;
          // we expect the unique_ptr to hold an object, exception should have been thrown
          // otherwise
          auto object = DataRefUtils::as<NonConstT>(ref);
          // need to swap the content of the deserialized container to a local variable to force return
          // value optimization
          T container;
          std::swap(const_cast<NonConstT&>(container), *object);
          return container;
        } else {
          throw std::runtime_error("Attempt to extract object from message with unsupported serialization type");
        }
      } else {
        static_assert(always_static_assert_v<T>, "unsupported code path");
      }

      // implementation (h)
    } else if constexpr (is_messageable<T>::value) {
      // extract a messageable type by reference
      // Cast content of payload bound by @a binding to known type.
      // we need to check the serialization type, the cast makes only sense for
      // unserialized objects
      using DataHeader = o2::header::DataHeader;

      auto&& ref = get<DataRef>(binding);
      auto header = o2::header::get<const DataHeader*>(ref.header);
      auto method = header->payloadSerializationMethod;
      if (method != o2::header::gSerializationMethodNone) {
        // FIXME: we could in principle support serialized content here as well if we
        // store all extracted objects internally and provide cleanup
        throw std::runtime_error("Can not extract a plain object from serialized message");
      }
      return *reinterpret_cast<T const*>(ref.payload);

      // implementation (i)
    } else if constexpr (std::is_pointer<T>::value &&
                         (is_messageable<typename std::remove_pointer<T>::type>::value || has_root_dictionary<typename std::remove_pointer<T>::type>::value)) {
      // extract a messageable type or object with ROOT dictionary by pointer
      // return unique_ptr to message content with custom deleter
      using DataHeader = o2::header::DataHeader;
      using ValueT = typename std::remove_pointer<T>::type;

      auto&& ref = get<DataRef>(binding);
      auto header = o2::header::get<const DataHeader*>(ref.header);
      auto method = header->payloadSerializationMethod;
      if (method == o2::header::gSerializationMethodNone) {
        if constexpr (is_messageable<ValueT>::value) {
          auto const* ptr = reinterpret_cast<ValueT const*>(ref.payload);
          // return type with non-owning Deleter instance
          std::unique_ptr<ValueT const, Deleter<ValueT const>> result(ptr, Deleter<ValueT const>(false));
          return result;
        } else if constexpr (is_specialization<ValueT, std::vector>::value && has_messageable_value_type<ValueT>::value) {
          // TODO: construct a vector spectator
          // this is a quick solution now which makes a copy of the plain vector data
          auto* start = reinterpret_cast<typename ValueT::value_type const*>(ref.payload);
          auto* end = start + header->payloadSize / sizeof(typename ValueT::value_type);
          auto container = std::make_unique<ValueT>(start, end);
          std::unique_ptr<ValueT const, Deleter<ValueT const>> result(container.release(), Deleter<ValueT const>(true));
          return result;
        }
        throw std::runtime_error("unsupported code path");
      } else if (method == o2::header::gSerializationMethodROOT) {
        // This supports the common case of retrieving a root object and getting pointer.
        // Notice that this will return a copy of the actual contents of the buffer, because
        // the buffer is actually serialised, for this reason we return a unique_ptr<T>.
        // FIXME: does it make more sense to keep ownership of all the deserialised
        // objects in a single place so that we can avoid duplicate deserializations?
        // explicitely specify serialization method to ROOT-serialized because type T
        // is messageable and a different method would be deduced in DataRefUtils
        // return type with owning Deleter instance, forwarding to default_deleter
        std::unique_ptr<ValueT const, Deleter<ValueT const>> result(DataRefUtils::as<ROOTSerialized<ValueT>>(ref).release());
        return result;
      } else {
        throw std::runtime_error("Attempt to extract object from message with unsupported serialization type");
      }
    } else if constexpr (has_root_dictionary<T>::value) {
      // retrieving ROOT objects follows the pointer approach, i.e. T* has to be specified
      // as template parameter and a unique_ptr will be returned, std vectors of ROOT serializable
      // objects can be retrieved by move, this is handled above in the "container" code branch
      static_assert(always_static_assert_v<T>, "ROOT objects need to be retrieved by pointer");
    } else {
      // non-messageable objects for which serialization method can not be derived by type,
      // the operation depends on the transmitted serialization method
      using DataHeader = o2::header::DataHeader;

      auto&& ref = get(binding);
      auto header = o2::header::get<const DataHeader*>(ref.header);
      auto method = header->payloadSerializationMethod;
      if (method == o2::header::gSerializationMethodNone) {
        // this code path is only selected if the type is non-messageable
        throw std::runtime_error(
          "Type mismatch: attempt to extract a non-messagable object "
          "from message with unserialized data");
      } else if (method == o2::header::gSerializationMethodROOT) {
        // explicitely specify serialization method to ROOT-serialized because type T
        // is messageable and a different method would be deduced in DataRefUtils
        // return type with owning Deleter instance, forwarding to default_deleter
        std::unique_ptr<T const, Deleter<T const>> result(DataRefUtils::as<ROOTSerialized<T>>(ref).release());
        return result;
      } else {
        throw std::runtime_error("Attempt to extract object from message with unsupported serialization type");
      }
    }
  }

  template <typename T = DataRef>
  decltype(auto) get(std::string const& binding) const
  {
    return get<T>(binding.c_str());
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
