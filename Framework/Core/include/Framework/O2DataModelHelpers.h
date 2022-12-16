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

#ifndef O2_FRAMEWORK_O2DATAMODELHELPERS_H_
#define O2_FRAMEWORK_O2DATAMODELHELPERS_H_

#include "Headers/DataHeader.h"
#include <fairmq/FairMQParts.h>
#include "Framework/OutputSpec.h"

namespace o2::framework
{
/// Set of helpers to deal with the O2 data model
/// in particular the way we deal with the split
/// payloads and the associated headers.
struct O2DataModelHelpers {
  /// Apply the function F to each header
  /// keeping into account the split payloads.
  template <typename F>
  static void for_each_header(fair::mq::Parts& parts, F&& f)
  {
    size_t i = 0;
    while (i < parts.Size()) {
      auto* dh = o2::header::get<o2::header::DataHeader*>(parts.At(i)->GetData());
      f(dh);
      // One for the header, plus the number of split parts.
      // When 0 it means we have a single part.
      i += 1 + (dh->splitPayloadParts ? dh->splitPayloadParts : 1);
    }
  }

  // Return true if F returns true for any of the headers
  template <typename F>
  static bool any_header_matching(fair::mq::Parts& parts, F&& f)
  {
    size_t i = 0;
    while (i < parts.Size()) {
      auto* dh = o2::header::get<o2::header::DataHeader*>(parts.At(i)->GetData());
      if (f(dh)) {
        return true;
      }
      // One for the header, plus the number of split parts.
      // When 0 it means we have a single part.
      i += 1 + (dh->splitPayloadParts ? dh->splitPayloadParts : 1);
    }
    return false;
  }

  // Return true if F returns true for any of the headers
  template <typename F>
  static bool all_headers_matching(fair::mq::Parts& parts, F&& f)
  {
    size_t i = 0;
    while (i < parts.Size()) {
      auto* dh = o2::header::get<o2::header::DataHeader*>(parts.At(i)->GetData());
      if (!f(dh)) {
        return false;
      }
      // One for the header, plus the number of split parts.
      // When 0 it means we have a single part.
      i += 1 + (dh->splitPayloadParts ? dh->splitPayloadParts : 1);
    }
    return true;
  }
  static void updateMissingSporadic(fair::mq::Parts& parts, std::vector<OutputSpec> const& specs, std::vector<bool>& present);
  static bool validateOutputs(std::vector<bool>& present)
  {
    for (auto p : present) {
      if (!p) {
        return false;
      }
    }
    return true;
  }
  static std::string describeMissingOutputs(std::vector<OutputSpec> const& specs, std::vector<bool> const& present);
};
} // namespace o2::framework

#endif // O2_FRAMEWORK_O2DATAMODELHELPERS_H_
