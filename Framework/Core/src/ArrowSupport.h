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
#ifndef O2_FRAMEWORK_ARROWSUPPORT_H_
#define O2_FRAMEWORK_ARROWSUPPORT_H_

#include "Framework/ServiceSpec.h"
#include "Framework/TypeIdHelpers.h"

namespace o2::framework
{

/// A few ServiceSpecs data sending backends
struct ArrowSupport {
  // Create spec for backend used to send Arrow messages
  static ServiceSpec arrowBackendSpec();
  static ServiceSpec arrowTableSlicingCacheDefSpec();
  static ServiceSpec arrowTableSlicingCacheSpec();
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_ARROWSUPPORT_H_
