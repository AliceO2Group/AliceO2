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
#include "Headers/DataHeader.h"
#include "Framework/TMessageSerializer.h"
#include <TClass.h>
#include <stdexcept>
#include <sstream>
#include <type_traits>
#include <gsl/gsl>

namespace o2 {
namespace framework {

// FIXME: Should enforce the fact that DataRefs are read only...
struct DataRefUtils {
  // SFINAE makes this available only for the case we are using
  // a POD, this is to distinguish it from the alternative below,
  // which works for TObject (which are serialised).
  template <typename T>
  static typename std::enable_if<std::is_pod<T>::value == true, gsl::span<T>>::type
  as(DataRef const &ref) {
    using DataHeader = o2::header::DataHeader;
    auto header = o2::header::get<const DataHeader>(ref.header);
    if (header->payloadSerializationMethod != o2::header::gSerializationMethodNone) {
      throw std::runtime_error("Attempt to extract a POD from a wrong message kind");
    }
    if ((header->payloadSize % sizeof(T)) != 0) {
      throw std::runtime_error("Cannot extract POD from message as size do not match");
    }
    //FIXME: provide a const collection
    return gsl::span<T>(reinterpret_cast<T *>(const_cast<char *>(ref.payload)), header->payloadSize/sizeof(T));
  }

  // See above. SFINAE allows us to use this to extract a TObject with 
  // a somewhat uniform API.
  template <typename T>
  static typename std::enable_if<std::is_base_of<TObject, T>::value == true, std::unique_ptr<T>>::type
  as(DataRef const &ref) {
    using DataHeader = o2::header::DataHeader;
    auto header = o2::header::get<const DataHeader>(ref.header);
    if (header->payloadSerializationMethod != o2::header::gSerializationMethodROOT) {
      throw std::runtime_error("Attempt to extract a TMessage from non-ROOT serialised message");
    }

    o2::framework::FairTMessage ftm(const_cast<char *>(ref.payload), header->payloadSize);
    auto cobj = ftm.ReadObject(ftm.GetClass());
    std::unique_ptr<T> result;
    result.reset(dynamic_cast<T*>(cobj));
    if (result.get() == nullptr) {
      std::ostringstream ss;
      ss << "Attempting to extract a " << T::Class()->GetName()
         << " but a " << cobj->ClassName()
         << " is actually stored which cannot be casted to the requested one.";
      throw std::runtime_error(ss.str());
    }
    return std::move(result);
  }
};

}
}

#endif // FRAMEWORK_DATAREFUTILS_H
