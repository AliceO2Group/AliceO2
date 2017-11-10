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
#include "Framework/Collection.h"
#include "Headers/DataHeader.h"

namespace o2 {
namespace framework {

// FIXME: Should enforce the fact that DataRefs are read only...
struct DataRefUtils {
  template <typename T>
  static Collection<T> as(const DataRef &ref) {
    using DataHeader = o2::Header::DataHeader;
    auto header = o2::Header::get<const DataHeader>(ref.header);
    assert((header->payloadSize % sizeof(T)) == 0);
    //FIXME: provide a const collection
    return Collection<T>(reinterpret_cast<void *>(const_cast<char *>(ref.payload)), header->payloadSize/sizeof(T));
  }
};

}
}

#endif // FRAMEWORK_DATAREFUTILS_H
