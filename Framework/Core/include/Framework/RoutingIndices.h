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

#ifndef O2_FRAMEWORK_ROUTINGINDICES_H_
#define O2_FRAMEWORK_ROUTINGINDICES_H_

namespace o2::framework
{

// An index in the space of the available Routes.
// This takes into account that you might have multiple
// routes (i.e. pipelined devices) which could provide
// the same kind of data.
struct RouteIndex {
  int value;
  explicit operator int() const { return value; }
};

// An index in the space of the declared InputSpec
// This does not take multiple input routes into account
struct InputIndex {
  int value;
  explicit operator int() const { return value; }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_ROUTINGINDICES_H_
