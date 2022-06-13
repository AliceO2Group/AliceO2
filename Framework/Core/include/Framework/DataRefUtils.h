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
#ifndef O2_FRAMEWORK_DATAREFUTILS_H_
#define O2_FRAMEWORK_DATAREFUTILS_H_

#include "Framework/DataRef.h"
#include "Framework/RootSerializationSupport.h"
#include "Framework/SerializationMethods.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/TypeTraits.h"
#include "Headers/DataHeader.h"
#include "Framework/CheckTypes.h"
#include "Framework/RuntimeError.h"

#include <gsl/gsl>

#include <type_traits>
#include <typeinfo>

namespace o2::conf
{
class ConfigurableParam;
}

namespace o2::framework
{

// FIXME: Should enforce the fact that DataRefs are read only...
struct DataRefUtils {

  template <typename T>
  static auto as(DataRef const& ref)
  {
    auto payloadSize = DataRefUtils::getPayloadSize(ref);
    // SFINAE makes this available only for the case we are using
    // trivially copyable type, this is to distinguish it from the
    // alternative below, which works for TObject (which are serialised).
    if constexpr (is_messageable<T>::value == true) {
      using DataHeader = o2::header::DataHeader;
      auto header = o2::header::get<const DataHeader*>(ref.header);
      if (header->payloadSerializationMethod != o2::header::gSerializationMethodNone) {
        throw runtime_error("Attempt to extract a POD from a wrong message kind");
      }
      if ((payloadSize % sizeof(T)) != 0) {
        throw runtime_error("Cannot extract POD from message as size do not match");
      }
      //FIXME: provide a const collection
      return gsl::span<T>(reinterpret_cast<T*>(const_cast<char*>(ref.payload)), payloadSize / sizeof(T));
    } else if constexpr (has_root_dictionary<T>::value == true &&
                         is_messageable<T>::value == false) {
      std::unique_ptr<T> result;
      static_assert(is_type_complete_v<struct RootSerializationSupport>, "Framework/RootSerializationSupport.h not included");
      call_if_defined<struct RootSerializationSupport>([&](auto* p) {
        // Classes using the ROOT ClassDef/ClassImp macros can be detected automatically
        // and the serialization method can be deduced. If the class is also
        // messageable, gSerializationMethodNone is used by default. This substitution
        // does not apply for such types, the serialization method needs to be specified
        // explicitely using type wrapper @a ROOTSerialized.
        using RSS = std::decay_t<decltype(*p)>;
        using DataHeader = o2::header::DataHeader;
        auto header = o2::header::get<const DataHeader*>(ref.header);
        if (header->payloadSerializationMethod != o2::header::gSerializationMethodROOT) {
          throw runtime_error("Attempt to extract a TMessage from non-ROOT serialised message");
        }

        typename RSS::FairTMessage ftm(const_cast<char*>(ref.payload), payloadSize);
        auto* storedClass = ftm.GetClass();
        auto* requestedClass = RSS::TClass::GetClass(typeid(T));
        // should always have the class description if has_root_dictionary is true
        assert(requestedClass != nullptr);

        auto* object = ftm.ReadObjectAny(storedClass);
        if (object == nullptr) {
          throw runtime_error_f("Failed to read object with name %s from message using ROOT serialization.",
                                (storedClass != nullptr ? storedClass->GetName() : "<unknown>"));
        }

        if constexpr (std::is_base_of<typename RSS::TObject, T>::value) { // compile time switch
          // if the type to be extracted inherits from TObject, the class descriptions
          // do not need to match exactly, only the dynamic_cast has to work
          // FIXME: could probably try to extend this to arbitrary types T being a base
          // to the stored type, but this would require to cast the extracted object to
          // the actual type. This requires this information to be available as a type
          // in the ROOT dictionary, check if this is the case or if TClass::DynamicCast
          // can be used for the test. Right now, this case was and is not covered and
          // not yet checked in the unit test either.
          auto* r = dynamic_cast<T*>(static_cast<typename RSS::TObject*>(object));
          if (r) {
            result.reset(r);
          }
          // This check includes the case: storedClass == requestedClass
        } else if (storedClass->InheritsFrom(requestedClass)) {
          result.reset(static_cast<T*>(object));
        }
        if (result == nullptr) {
          // did not manage to cast the pointer to result
          // delete object via the class info if available, not the case for all types,
          // e.g. for standard containers of ROOT objects apparently this is not always
          // the case
          auto* delfunc = storedClass->GetDelete();
          if (delfunc) {
            (*delfunc)(object);
          }

          throw runtime_error_f("Attempting to extract a %s but a %s is actually stored which cannot be casted to the requested one.",
                                (requestedClass != nullptr ? requestedClass->GetName() : "<unknown>"),
                                (storedClass != nullptr ? storedClass->GetName() : "<unknown>"));
        }
        // collections in ROOT can be non-owning or owning and the proper cleanup depends on
        // this flag. Be it a bug or a feature in ROOT, but the owning flag of the extracted
        // object only depends on the state at serialization of the original object. However,
        // all objects created during deserialization are new and must be owned by the collection
        // to avoid memory leak. So we call SetOwner if it is available for the type.
        if constexpr (has_root_setowner<T>::value) {
          result->SetOwner(true);
        }
      });

      return std::move(result);
    } else if constexpr (is_specialization_v<T, ROOTSerialized> == true) {
      // See above. SFINAE allows us to use this to extract a ROOT-serialized object
      // with a somewhat uniform API. ROOT serialization method is enforced by using
      // type wrapper @a ROOTSerialized
      static_assert(is_type_complete_v<struct RootSerializationSupport>, "Framework/RootSerializationSupport.h not included");
      using wrapped = typename T::wrapped_type;
      using DataHeader = o2::header::DataHeader;
      std::unique_ptr<wrapped> result;

      call_if_defined<struct RootSerializationSupport>([&](auto* p) {
        using RSS = std::decay_t<decltype(*p)>;

        auto header = o2::header::get<const DataHeader*>(ref.header);
        if (header->payloadSerializationMethod != o2::header::gSerializationMethodROOT) {
          throw runtime_error("Attempt to extract a TMessage from non-ROOT serialised message");
        }
        auto* cl = RSS::TClass::GetClass(typeid(wrapped));
        if (has_root_dictionary<wrapped>::value == false && cl == nullptr) {
          throw runtime_error("ROOT serialization not supported, dictionary not found for data type");
        }

        typename RSS::FairTMessage ftm(const_cast<char*>(ref.payload), payloadSize);
        result.reset(static_cast<wrapped*>(ftm.ReadObjectAny(cl)));
        if (result.get() == nullptr) {
          throw runtime_error_f("Unable to extract class %s", cl == nullptr ? "<name not available>" : cl->GetName());
        }
        // workaround for ROOT feature, see above
        if constexpr (has_root_setowner<T>::value) {
          result->SetOwner(true);
        }
      });
      return std::move(result);
    } else if constexpr (is_specialization_v<T, CCDBSerialized> == true) {
      using wrapped = typename T::wrapped_type;
      using DataHeader = o2::header::DataHeader;
      auto* ptr = DataRefUtils::decodeCCDB(ref, typeid(wrapped));
      if constexpr (std::is_base_of<o2::conf::ConfigurableParam, wrapped>::value) {
        auto& param = const_cast<typename std::remove_const<wrapped&>::type>(wrapped::Instance());
        param.syncCCDBandRegistry(ptr);
        ptr = &param;
      }
      std::unique_ptr<wrapped> result(static_cast<wrapped*>(ptr));
      return std::move(result);
    }
  }
  // Decode a CCDB object using the CcdbApi.
  static void* decodeCCDB(DataRef const& ref, std::type_info const& info);

  static o2::header::DataHeader::PayloadSizeType getPayloadSize(const DataRef& ref)
  {
    using DataHeader = o2::header::DataHeader;
    auto* header = o2::header::get<const DataHeader*>(ref.header);
    if (!header) {
      return 0;
    }
    // in case of an O2 message with multiple payloads, the size of the message stored
    // in DataRef is returned,
    // as a prototype solution we are using splitPayloadIndex == splitPayloadParts to
    // indicate that there are splitPayloadParts payloads following the header
    if (header->splitPayloadParts > 1 && header->splitPayloadIndex == header->splitPayloadParts) {
      return ref.payloadSize;
    }
    return header->payloadSize < ref.payloadSize || ref.payloadSize == 0 ? header->payloadSize : ref.payloadSize;
  }

  template <typename T>
  static auto getHeader(const DataRef& ref)
  {
    using HeaderT = typename std::remove_pointer<T>::type;
    static_assert(std::is_pointer<T>::value && std::is_base_of<o2::header::BaseHeader, HeaderT>::value,
                  "pointer to BaseHeader-derived type required");
    return o2::header::get<T>(ref.header);
  }

  static bool isValid(DataRef const& ref)
  {
    return ref.header != nullptr;
  }

  /// check if the DataRef object matches a partcular binding
  /// Can be used to check if DataRef from the InputRecord iterator comes from
  /// a particular input.
  static bool match(DataRef const& ref, const char* binding)
  {
    return ref.spec != nullptr && ref.spec->binding == binding;
  }

  /// check if the O2 message referred by DataRef matches a particular
  /// input spec. The DataHeader is retrieved from the header message and matched
  /// against @ref spec parameter.
  static bool match(DataRef const& ref, InputSpec const& spec)
  {
    auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    if (dh == nullptr) {
      return false;
    }
    return DataSpecUtils::match(spec, dh->dataOrigin, dh->dataDescription, dh->subSpecification);
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATAREFUTILS_H_
