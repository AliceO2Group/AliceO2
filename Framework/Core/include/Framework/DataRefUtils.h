// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAREFUTILS_H
#define FRAMEWORK_DATAREFUTILS_H

#include "Framework/DataRef.h"
#include "Framework/TMessageSerializer.h"
#include "Framework/SerializationMethods.h"
#include "Framework/TypeTraits.h"

#include "Headers/DataHeader.h"

#include <TClass.h>
#include <gsl/gsl>

#include <stdexcept>
#include <sstream>
#include <type_traits>

namespace o2
{
namespace framework
{

// FIXME: Should enforce the fact that DataRefs are read only...
struct DataRefUtils {
  // SFINAE makes this available only for the case we are using
  // trivially copyable type, this is to distinguish it from the
  // alternative below, which works for TObject (which are serialised).
  template <typename T>
  static typename std::enable_if<is_messageable<T>::value == true, gsl::span<T>>::type
    as(DataRef const& ref)
  {
    using DataHeader = o2::header::DataHeader;
    auto header = o2::header::get<const DataHeader*>(ref.header);
    if (header->payloadSerializationMethod != o2::header::gSerializationMethodNone) {
      throw std::runtime_error("Attempt to extract a POD from a wrong message kind");
    }
    if ((header->payloadSize % sizeof(T)) != 0) {
      throw std::runtime_error("Cannot extract POD from message as size do not match");
    }
    //FIXME: provide a const collection
    return gsl::span<T>(reinterpret_cast<T*>(const_cast<char*>(ref.payload)), header->payloadSize / sizeof(T));
  }

  // See above. SFINAE allows us to use this to extract a ROOT-serialized object
  // with a somewhat uniform API.
  // Classes using the ROOT ClassDef/ClassImp macros can be detected automatically
  // and the serialization method can be deduced. If the class is also
  // messageable, gSerializationMethodNone is used by default. This substitution
  // does not apply for such types, the serialization method needs to be specified
  // explicitely using type wrapper @a ROOTSerialized.
  template <typename T>
  static typename std::enable_if<has_root_dictionary<T>::value == true &&
                                   is_messageable<T>::value == false,
                                 std::unique_ptr<T>>::type
    as(DataRef const& ref)
  {
    using DataHeader = o2::header::DataHeader;
    auto header = o2::header::get<const DataHeader*>(ref.header);
    if (header->payloadSerializationMethod != o2::header::gSerializationMethodROOT) {
      throw std::runtime_error("Attempt to extract a TMessage from non-ROOT serialised message");
    }

    o2::framework::FairTMessage ftm(const_cast<char*>(ref.payload), header->payloadSize);
    auto* storedClass = ftm.GetClass();
    auto* requestedClass = TClass::GetClass(typeid(T));
    // should always have the class description if has_root_dictionary is true
    assert(requestedClass != nullptr);

    auto* object = ftm.ReadObjectAny(storedClass);
    if (object == nullptr) {
      std::ostringstream ss;
      ss << "Failed to read object with name "
         << (storedClass != nullptr ? storedClass->GetName() : "<unknown>")
         << " from message using ROOT serialization.";
      throw std::runtime_error(ss.str());
    }

    std::unique_ptr<T> result;
    if (std::is_base_of<TObject, T>::value) { // compile time switch
      // if the type to be extracted inherits from TObject, the class descriptions
      // do not need to match exactly, only the dynamic_cast has to work
      // FIXME: could probably try to extend this to arbitrary types T being a base
      // to the stored type, but this would require to cast the extracted object to
      // the actual type. This requires this information to be available as a type
      // in the ROOT dictionary, check if this is the case or if TClass::DynamicCast
      // can be used for the test. Right now, this case was and is not covered and
      // not yet checked in the unit test either.
      auto* r = dynamic_cast<T*>(static_cast<TObject*>(object));
      if (r) {
        result.reset(r);
      }
    } else if (storedClass == requestedClass) {
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

      std::ostringstream ss;
      ss << "Attempting to extract a "
         << (requestedClass != nullptr ? requestedClass->GetName() : "<unknown>")
         << " but a "
         << (storedClass != nullptr ? storedClass->GetName() : "<unknown>")
         << " is actually stored which cannot be casted to the requested one.";
      throw std::runtime_error(ss.str());
    }

    return std::move(result);
  }

  // See above. SFINAE allows us to use this to extract a ROOT-serialized object
  // with a somewhat uniform API. ROOT serialization method is enforced by using
  // type wrapper @a ROOTSerialized
  template <typename W>
  static typename std::enable_if<is_specialization<W, ROOTSerialized>::value == true,
                                 std::unique_ptr<typename W::wrapped_type>>::type
    as(DataRef const& ref)
  {
    using T = typename W::wrapped_type;
    using DataHeader = o2::header::DataHeader;
    auto header = o2::header::get<const DataHeader*>(ref.header);
    if (header->payloadSerializationMethod != o2::header::gSerializationMethodROOT) {
      throw std::runtime_error("Attempt to extract a TMessage from non-ROOT serialised message");
    }
    auto* cl = TClass::GetClass(typeid(T));
    if (has_root_dictionary<T>::value == false && cl == nullptr) {
      throw std::runtime_error("ROOT serialization not supported, dictionary not found for data type");
    }

    o2::framework::FairTMessage ftm(const_cast<char*>(ref.payload), header->payloadSize);
    std::unique_ptr<T> result(static_cast<T*>(ftm.ReadObjectAny(cl)));
    if (result.get() == nullptr) {
      std::ostringstream ss;
      ss << "Unable to extract class ";
      if (cl == nullptr) {
        ss << "<name not available>";
      } else {
        ss << cl->GetName();
      }
      throw std::runtime_error(ss.str());
    }
    return std::move(result);
  }

  static unsigned getPayloadSize(const DataRef& ref)
  {
    using DataHeader = o2::header::DataHeader;
    auto* header = o2::header::get<const DataHeader*>(ref.header);
    if (!header) {
      return 0;
    }
    return header->payloadSize;
  }

  template <typename T>
  static auto getHeader(const DataRef& ref)
  {
    using HeaderT = typename std::remove_pointer<T>::type;
    static_assert(std::is_pointer<T>::value && std::is_base_of<o2::header::BaseHeader, HeaderT>::value,
                  "pointer to BaseHeader-derived type required");
    return o2::header::get<T>(ref.header);
  }
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DATAREFUTILS_H
