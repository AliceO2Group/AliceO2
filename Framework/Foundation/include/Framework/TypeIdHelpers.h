// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_TYPEIDHELPERS_H_
#define O2_FRAMEWORK_TYPEIDHELPERS_H_

#include "Framework/StringHelpers.h"

namespace o2::framework
{

struct TypeIdHelpers {
  /// Return a unique id for a given type
  /// This works just fine with GCC and CLANG,
  /// C++20 will allow us to use:
  ///    std::experimental::source_location::current().function_name();
  template <typename T>
  constexpr static uint32_t uniqueId()
  {
    return compile_time_hash(__PRETTY_FUNCTION__);
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_TYPEIDHELPERS_H_
