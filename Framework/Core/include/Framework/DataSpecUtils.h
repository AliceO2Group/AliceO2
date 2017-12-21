// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATASPECUTILS_H
#define FRAMEWORK_DATASPECUTILS_H

#include "Framework/OutputSpec.h"
#include "Framework/InputSpec.h"
#include "Headers/DataHeader.h"

namespace o2 {
namespace framework {

struct DataSpecUtils {
  static bool match(const InputSpec &spec,
                    const o2::header::DataOrigin &origin,
                    const o2::header::DataDescription &description,
                    const o2::header::DataHeader::SubSpecificationType &subSpec) {
    return spec.origin == origin &&
           spec.description == description &&
           spec.subSpec == subSpec;
  }

  static bool match(const OutputSpec &spec,
                    const o2::header::DataOrigin &origin,
                    const o2::header::DataDescription &description,
                    const o2::header::DataHeader::SubSpecificationType &subSpec) {
    return spec.origin == origin &&
           spec.description == description &&
           spec.subSpec == subSpec;
  }

  template <typename T>
  static bool match(const T&spec, const o2::header::DataHeader &header) {
    return DataSpecUtils::match(spec,
                                header.dataOrigin,
                                header.dataDescription,
                                header.subSpecification);
  }

};
} // namespace framework
} // namespace o2
#endif // FRAMEWORK_DATASPECUTILS_H
