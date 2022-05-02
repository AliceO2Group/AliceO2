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
#ifndef O2_FRAMEWORK_INPUTRECORD_H_
#define O2_FRAMEWORK_INPUTRECORD_H_

#include "Framework/DataRef.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRoute.h"
#include "Framework/TypeTraits.h"
#include "Framework/TableConsumer.h"
#include "Framework/Traits.h"
#include "Framework/RuntimeError.h"
#include "Framework/Logger.h"
#include "Framework/ObjectCache.h"
#include "Framework/CallbackService.h"

#include "Headers/DataHeader.h"

#include "CommonUtils/BoostSerializer.h"

#include <gsl/gsl>

#include <iterator>
#include <string>
#include <vector>
#include <cstring>
#include <cassert>
#include <memory>
#include <type_traits>

#include <fairmq/FwdDecls.h>

namespace o2::framework
{

struct InputSpec;
class InputSpan;
class CallbackService;

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

  // Typesafe position inside a record of an input.
  // Multiple routes by which the input gets in this
  // position are multiplexed.
  struct InputPos {
    size_t index;
    constexpr static size_t INVALID = -1LL;
  };

  InputRecord(std::vector<InputRoute> const& inputs,
              InputSpan& span,
              ServiceRegistry&);

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
    constexpr Deleter(const self_type& other) : base::default_delete(other), mProperty{OwnershipProperty::Unknown}
    {
      if (mProperty == OwnershipProperty::Unknown) {
        mProperty = other.mProperty;
      } else if (mProperty != other.mProperty) {
        throw runtime_error("Attemp to change resource control");
      }
    }

    // copy constructor for the default delete which simply sets the
    // resource ownership control to 'Owning'
    constexpr Deleter(const base& other) : base::default_delete(other), mProperty{OwnershipProperty::Owning} {}

    // allow assignment operator only for pristine or matching resource control property
    self_type& operator=(const self_type& other)
    {
      // the default_deleter does not have any state, so this could be skipped, but keep the call to
      // the base for completeness, and the (small) chance for changing the base
      base::operator=(other);
      if (mProperty == OwnershipProperty::Unknown) {
        mProperty = other.mProperty;
      } else if (mProperty != other.mProperty) {
        throw runtime_error("Attemp to change resource control");
      }
      return *this;
    }

    void operator()(T* ptr) const
    {
      if (mProperty == OwnershipProperty::NotOwning) {
        // nothing done if resource is not owned
        return;
      }
      base::operator()(ptr);
    }

   private:
    OwnershipProperty mProperty = OwnershipProperty::Unknown;
  };

  int getPos(const char* name) const;
  [[nodiscard]] static InputPos getPos(std::vector<InputRoute> const& routes, ConcreteDataMatcher matcher);
  [[nodiscard]] static DataRef getByPos(std::vector<InputRoute> const& routes, InputSpan const& span, int pos, int part = 0);

  [[nodiscard]] int getPos(const std::string& name) const;

  [[nodiscard]] DataRef getByPos(int pos, int part = 0) const;

  /// Get the ref of the first valid input. If requested, throw an error if none is found.
  [[nodiscard]] DataRef getFirstValid(bool throwOnFailure = false) const;

  [[nodiscard]] size_t getNofParts(int pos) const;
  /// Get the object of specified type T for the binding R.
  /// If R is a string like object, we look up by name the InputSpec and
  /// return the data associated to the given label.
  /// If R is a DataRef, we extract the result object from the Payload,
  /// following the information provided by the Header.
  /// The actual operation and cast depends on the target data type and the
  /// serialization type of the incoming data.
  /// By default we return a DataRef, which is the pair of pointers to
  /// the header and payload of the O2 Message.
  /// See @ref Inputrecord class description for supported types.
  /// @param ref   DataRef with pointers to input spec, header, and payload
  template <typename T = DataRef, typename R>
  decltype(auto) get(R binding, int part = 0) const
  {
    DataRef ref{nullptr, nullptr};
    using decayed = std::decay_t<R>;
    // Get the actual dataref
    if constexpr (std::is_same_v<decayed, char const*> ||
                  std::is_same_v<decayed, char*> ||
                  std::is_same_v<decayed, std::string>) {
      try {
        int pos = -1;
        if constexpr (std::is_same_v<decayed, std::string>) {
          pos = getPos(binding.c_str());
        } else {
          pos = getPos(binding);
        }
        if (pos < 0) {
          throw std::invalid_argument("no matching route found for " + std::string(binding));
        }
        ref = this->getByPos(pos, part);
      } catch (const std::exception& e) {
        if constexpr (std::is_same_v<decayed, std::string>) {
          throw runtime_error_f("Unknown argument requested %s - %s", binding.c_str(), e.what());
        } else {
          throw runtime_error_f("Unknown argument requested %s - %s", binding, e.what());
        }
      }
    } else if constexpr (std::is_same_v<decayed, DataRef>) {
      ref = binding;
    } else {
      static_assert(always_static_assert_v<R>, "Unknown binding type");
    }

    using PointerLessValueT = std::remove_pointer_t<T>;

    if constexpr (std::is_same_v<std::decay_t<T>, DataRef>) {
      return ref;
    } else if constexpr (std::is_same<T, std::string>::value) {
      // substitution for std::string
      // If we ask for a string, we need to duplicate it because we do not want
      // the buffer to be deleted when it goes out of scope. The string is built
      // from the data and its lengh, null-termination is not necessary.
      // return std::string object
      return std::string(ref.payload, DataRefUtils::getPayloadSize(ref));

      // implementation (c)
    } else if constexpr (std::is_same<T, char const*>::value) {
      // substitution for const char*
      // If we ask for a char const *, we simply point to the payload. Notice this
      // is meant for C-style strings which are expected to be null terminated.
      // If you want to actually get hold of the buffer, use gsl::span<char> as that will
      // give you the size as well.
      // return pointer to payload content
      return reinterpret_cast<char const*>(ref.payload);

      // implementation (d)
    } else if constexpr (std::is_same<T, TableConsumer>::value) {
      // substitution for TableConsumer
      // For the moment this is dummy, as it requires proper support to
      // create the RDataSource from the arrow buffer.
      auto data = reinterpret_cast<uint8_t const*>(ref.payload);
      return std::make_unique<TableConsumer>(data, DataRefUtils::getPayloadSize(ref));

      // implementation (e)
    } else if constexpr (framework::is_boost_serializable<T>::value || is_specialization_v<T, BoostSerialized>) {
      // substitution for boost-serialized entities
      // We have to deserialize the ostringstream.
      // FIXME: check that the string is null terminated.
      // @return deserialized copy of payload
      auto str = std::string(ref.payload, DataRefUtils::getPayloadSize(ref));
      assert(DataRefUtils::getPayloadSize(ref) == sizeof(T));
      if constexpr (is_specialization_v<T, BoostSerialized>) {
        return o2::utils::BoostDeserialize<typename T::wrapped_type>(str);
      } else {
        return o2::utils::BoostDeserialize<T>(str);
      }

      // implementation (f)
    } else if constexpr (is_span<T>::value) {
      // substitution for span of messageable objects
      // FIXME: there will be std::span in C++20
      static_assert(is_messageable<typename T::value_type>::value, "span can only be created for messageable types");
      auto header = DataRefUtils::getHeader<header::DataHeader*>(ref);
      assert(header);
      if (sizeof(typename T::value_type) > 1 && header->payloadSerializationMethod != o2::header::gSerializationMethodNone) {
        throw runtime_error("Inconsistent serialization method for extracting span");
      }
      using ValueT = typename T::value_type;
      auto payloadSize = DataRefUtils::getPayloadSize(ref);
      if (payloadSize % sizeof(ValueT)) {
        throw runtime_error(("Inconsistent type and payload size at " + std::string(ref.spec->binding) + "(" + DataSpecUtils::describe(*ref.spec) + ")" +
                             ": type size " + std::to_string(sizeof(ValueT)) +
                             "  payload size " + std::to_string(payloadSize))
                              .c_str());
      }
      return gsl::span<ValueT const>(reinterpret_cast<ValueT const*>(ref.payload), payloadSize / sizeof(ValueT));

      // implementation (g)
    } else if constexpr (is_container<T>::value) {
      // currently implemented only for vectors
      if constexpr (is_specialization_v<std::remove_const_t<T>, std::vector>) {
        auto header = DataRefUtils::getHeader<header::DataHeader*>(ref);
        auto payloadSize = DataRefUtils::getPayloadSize(ref);
        auto method = header->payloadSerializationMethod;
        if (method == o2::header::gSerializationMethodNone) {
          // TODO: construct a vector spectator
          // this is a quick solution now which makes a copy of the plain vector data
          auto* start = reinterpret_cast<typename T::value_type const*>(ref.payload);
          auto* end = start + payloadSize / sizeof(typename T::value_type);
          T result(start, end);
          return result;
        } else if (method == o2::header::gSerializationMethodROOT) {
          /// substitution for container of non-messageable objects with ROOT dictionary
          /// Notice that this will return a copy of the actual contents of the buffer, because
          /// the buffer is actually serialised. The extracted container is swaped to local,
          /// container, C++11 and beyond will implicitly apply return value optimization.
          /// @return std container object
          using NonConstT = typename std::remove_const<T>::type;
          if constexpr (is_specialization_v<T, ROOTSerialized> == true || has_root_dictionary<T>::value == true) {
            // we expect the unique_ptr to hold an object, exception should have been thrown
            // otherwise
            auto object = DataRefUtils::as<NonConstT>(ref);
            // need to swap the content of the deserialized container to a local variable to force return
            // value optimization
            T container;
            std::swap(const_cast<NonConstT&>(container), *object);
            return container;
          } else {
            throw runtime_error("No supported conversion function for ROOT serialized message");
          }
        } else {
          throw runtime_error("Attempt to extract object from message with unsupported serialization type");
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

      auto header = DataRefUtils::getHeader<header::DataHeader*>(ref);
      auto method = header->payloadSerializationMethod;
      if (method != o2::header::gSerializationMethodNone) {
        // FIXME: we could in principle support serialized content here as well if we
        // store all extracted objects internally and provide cleanup
        throw runtime_error("Can not extract a plain object from serialized message");
      }
      return *reinterpret_cast<T const*>(ref.payload);

      // implementation (i)
    } else if constexpr (std::is_pointer_v<T> &&
                         (is_messageable<PointerLessValueT>::value ||
                          has_root_dictionary<PointerLessValueT>::value ||
                          (is_specialization_v<PointerLessValueT, std::vector> && has_messageable_value_type<PointerLessValueT>::value) ||
                          (has_root_dictionary_mapped_type<PointerLessValueT>::value))) {
      // extract a messageable type or object with ROOT dictionary by pointer
      // return unique_ptr to message content with custom deleter
      using ValueT = PointerLessValueT;

      auto header = DataRefUtils::getHeader<header::DataHeader*>(ref);
      auto payloadSize = DataRefUtils::getPayloadSize(ref);
      auto method = header->payloadSerializationMethod;
      if (method == o2::header::gSerializationMethodNone) {
        if constexpr (is_messageable<ValueT>::value) {
          auto const* ptr = reinterpret_cast<ValueT const*>(ref.payload);
          // return type with non-owning Deleter instance
          std::unique_ptr<ValueT const, Deleter<ValueT const>> result(ptr, Deleter<ValueT const>(false));
          return result;
        } else if constexpr (is_specialization_v<ValueT, std::vector> && has_messageable_value_type<ValueT>::value) {
          // TODO: construct a vector spectator
          // this is a quick solution now which makes a copy of the plain vector data
          auto* start = reinterpret_cast<typename ValueT::value_type const*>(ref.payload);
          auto* end = start + payloadSize / sizeof(typename ValueT::value_type);
          auto container = std::make_unique<ValueT>(start, end);
          std::unique_ptr<ValueT const, Deleter<ValueT const>> result(container.release(), Deleter<ValueT const>(true));
          return result;
        }
        throw runtime_error("unsupported code path");
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
      } else if (method == o2::header::gSerializationMethodCCDB) {
        // This is to support deserialising objects from CCDB. Contrary to what happens for
        // other objects, those objects are most likely long lived, so we
        // keep around an instance of the associated object and deserialise it only when
        // it's updated.
        // FIXME: add ability to apply callbacks to deserialised objects.
        auto id = ObjectCache::Id::fromRef(ref);
        ConcreteDataMatcher matcher{header->dataOrigin, header->dataDescription, header->subSpecification};
        // If the matcher does not have an entry in the cache, deserialise it
        // and cache the deserialised object at the given id.
        auto path = fmt::format("{}", DataSpecUtils::describe(matcher));
        LOGP(info, "{}", path);
        auto& cache = mRegistry.get<ObjectCache>();
        auto& callbacks = mRegistry.get<CallbackService>();
        auto cacheEntry = cache.matcherToId.find(path);
        if (cacheEntry == cache.matcherToId.end()) {
          cache.matcherToId.insert(std::make_pair(path, id));
          std::unique_ptr<ValueT const, Deleter<ValueT const>> result(DataRefUtils::as<CCDBSerialized<ValueT>>(ref).release(), false);
          void* obj = (void*)result.get();
          callbacks(CallbackService::Id::CCDBDeserialised, (ConcreteDataMatcher&)matcher, (void*)obj);
          cache.idToObject[id] = obj;
          LOGP(info, "Caching in {} ptr to {} ({})", id.value, path, obj);
          return result;
        }
        auto& oldId = cacheEntry->second;
        // The id in the cache is the same, let's simply return it.
        if (oldId.value == id.value) {
          std::unique_ptr<ValueT const, Deleter<ValueT const>> result((ValueT const*)cache.idToObject[id], false);
          LOGP(info, "Returning cached entry {} for {} ({})", id.value, path, (void*)result.get());
          return result;
        }
        // The id in the cache is different. Let's destroy the old cached entry
        // and create a new one.
        delete reinterpret_cast<ValueT*>(cache.idToObject[oldId]);
        std::unique_ptr<ValueT const, Deleter<ValueT const>> result(DataRefUtils::as<CCDBSerialized<ValueT>>(ref).release(), false);
        void* obj = (void*)result.get();
        callbacks(CallbackService::Id::CCDBDeserialised, (ConcreteDataMatcher&)matcher, (void*)obj);
        cache.idToObject[id] = obj;
        LOGP(info, "Replacing cached entry {} with {} for {} ({})", oldId.value, id.value, path, obj);
        oldId.value = id.value;
        return result;
      } else {
        throw runtime_error("Attempt to extract object from message with unsupported serialization type");
      }
    } else if constexpr (std::is_pointer_v<T>) {
      static_assert(always_static_assert<T>::value, "T is not a supported type");
    } else if constexpr (has_root_dictionary<T>::value) {
      // retrieving ROOT objects follows the pointer approach, i.e. T* has to be specified
      // as template parameter and a unique_ptr will be returned, std vectors of ROOT serializable
      // objects can be retrieved by move, this is handled above in the "container" code branch
      static_assert(always_static_assert_v<T>, "ROOT objects need to be retrieved by pointer");
    } else {
      // non-messageable objects for which serialization method can not be derived by type,
      // the operation depends on the transmitted serialization method
      auto header = DataRefUtils::getHeader<header::DataHeader*>(ref);
      auto method = header->payloadSerializationMethod;
      if (method == o2::header::gSerializationMethodNone) {
        // this code path is only selected if the type is non-messageable
        throw runtime_error(
          "Type mismatch: attempt to extract a non-messagable object "
          "from message with unserialized data");
      } else if (method == o2::header::gSerializationMethodROOT) {
        // explicitely specify serialization method to ROOT-serialized because type T
        // is messageable and a different method would be deduced in DataRefUtils
        // return type with owning Deleter instance, forwarding to default_deleter
        std::unique_ptr<T const, Deleter<T const>> result(DataRefUtils::as<ROOTSerialized<T>>(ref).release());
        return result;
      } else {
        throw runtime_error("Attempt to extract object from message with unsupported serialization type");
      }
    }
  }

  template <typename T>
  T get_boost(char const* binding) const
  {
    DataRef ref = get<DataRef>(binding);
    auto str = std::string(ref.payload, DataRefUtils::getPayloadSize(ref));
    auto desData = o2::utils::BoostDeserialize<T>(str);
    return std::move(desData);
  }

  /// Helper method to be used to check if a given part of the InputRecord is present.
  [[nodiscard]] bool isValid(std::string const& s) const
  {
    return isValid(s.c_str());
  }

  /// Helper method to be used to check if a given part of the InputRecord is present.
  bool isValid(char const* s) const;
  [[nodiscard]] bool isValid(int pos) const;

  /// @return the total number of inputs in the InputRecord. Notice that these will include
  /// both valid and invalid inputs (i.e. inputs which have not arrived yet), depending
  /// on the CompletionPolicy you have (using the default policy all inputs will be valid).
  [[nodiscard]] size_t size() const;

  /// @return the total number of valid inputs in the InputRecord.
  /// Invalid inputs might happen if the CompletionPolicy allows
  /// incomplete records to be consumed or processed.
  [[nodiscard]] size_t countValidInputs() const;

  template <typename ParentT, typename T>
  class Iterator
  {
   public:
    using ParentType = ParentT;
    using SelfType = Iterator;
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using reference = T&;
    using pointer = T*;
    using difference_type = std::ptrdiff_t;
    using ElementType = typename std::remove_const<value_type>::type;

    Iterator() = delete;

    Iterator(ParentType const* parent, size_t position = 0, size_t size = 0)
      : mPosition(position), mSize(size > position ? size : position), mParent(parent), mElement{nullptr, nullptr, nullptr}
    {
      if (mPosition < mSize) {
        if (mParent->isValid(mPosition)) {
          mElement = mParent->getByPos(mPosition);
        } else {
          ++(*this);
        }
      }
    }

    ~Iterator() = default;

    // prefix increment
    SelfType& operator++()
    {
      while (mPosition < mSize && ++mPosition < mSize) {
        if (!mParent->isValid(mPosition)) {
          continue;
        }
        mElement = mParent->getByPos(mPosition);
        break;
      }
      if (mPosition >= mSize) {
        // reset the element to the default value of the type
        mElement = ElementType{};
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
    reference operator*() const
    {
      return mElement;
    }
    // comparison
    bool operator==(const SelfType& rh) const
    {
      return mPosition == rh.mPosition;
    }
    // comparison
    bool operator!=(const SelfType& rh) const
    {
      return mPosition != rh.mPosition;
    }

    [[nodiscard]] bool matches(o2::header::DataHeader matcher) const
    {
      if (mPosition >= mSize || mElement.header == nullptr) {
        return false;
      }
      // at this point there must be a DataHeader, this has been checked by the DPL
      // input cache
      const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(mElement);
      return *dh == matcher;
    }

    [[nodiscard]] bool matches(o2::header::DataOrigin origin, o2::header::DataDescription description = o2::header::gDataDescriptionInvalid) const
    {
      if (mPosition >= mSize || mElement.header == nullptr) {
        return false;
      }
      // at this point there must be a DataHeader, this has been checked by the DPL
      // input cache
      const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(mElement);
      return dh->dataOrigin == origin && (description == o2::header::gDataDescriptionInvalid || dh->dataDescription == description);
    }

    [[nodiscard]] bool matches(o2::header::DataOrigin origin, o2::header::DataDescription description, o2::header::DataHeader::SubSpecificationType subspec) const
    {
      return matches(o2::header::DataHeader{description, origin, subspec});
    }

    [[nodiscard]] ParentType const* parent() const
    {
      return mParent;
    }

    [[nodiscard]] size_t position() const
    {
      return mPosition;
    }

   private:
    size_t mPosition;
    size_t mSize;
    ParentType const* mParent;
    ElementType mElement;
  };

  /// @class InputRecordIterator
  /// An iterator over the input slots
  /// It supports an iterator interface to access the parts in the slot
  template <typename T>
  class InputRecordIterator : public Iterator<InputRecord, T>
  {
   public:
    using SelfType = InputRecordIterator;
    using BaseType = Iterator<InputRecord, T>;
    using value_type = typename BaseType::value_type;
    using reference = typename BaseType::reference;
    using pointer = typename BaseType::pointer;
    using ElementType = typename std::remove_const<value_type>::type;
    using iterator = Iterator<SelfType, T>;
    using const_iterator = Iterator<SelfType, const T>;

    InputRecordIterator(InputRecord const* parent, size_t position = 0, size_t size = 0)
      : BaseType(parent, position, size)
    {
    }

    /// Get element at {slotindex, partindex}
    [[nodiscard]] ElementType getByPos(size_t pos) const
    {
      return this->parent()->getByPos(this->position(), pos);
    }

    /// Check if slot is valid, index of part is not used
    [[nodiscard]] bool isValid(size_t = 0) const
    {
      if (this->position() < this->parent()->size()) {
        return this->parent()->isValid(this->position());
      }
      return false;
    }

    /// Get number of parts in input slot
    [[nodiscard]] size_t size() const
    {
      return this->parent()->getNofParts(this->position());
    }

    [[nodiscard]] const_iterator begin() const
    {
      return const_iterator(this, 0, size());
    }

    [[nodiscard]] const_iterator end() const
    {
      return const_iterator(this, size());
    }
  };

  using iterator = InputRecordIterator<DataRef>;
  using const_iterator = InputRecordIterator<const DataRef>;

  [[nodiscard]] const_iterator begin() const
  {
    return {this, 0, size()};
  }

  [[nodiscard]] const_iterator end() const
  {
    return {this, size()};
  }

  InputSpan& span()
  {
    return mSpan;
  }

 private:
  ServiceRegistry& mRegistry;
  std::vector<InputRoute> const& mInputsSchema;
  InputSpan& mSpan;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_INPUTREGISTRY_H_
