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

#ifndef O2_FRAMEWORK_TIMINGINFO_H_
#define O2_FRAMEWORK_TIMINGINFO_H_

#include <cstddef>
#include <cstdint>

/// This class holds the information about timing
/// of the messages being processed.
namespace o2::framework
{

struct TimingInfo {
  size_t timeslice; /// the timeslice associated to current processing
  uint32_t firstTForbit = -1; /// the orbit the TF begins
  uint32_t tfCounter = -1;    // the counter associated to a TF
  uint32_t runNumber = -1;
  uint64_t creation = -1UL;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_TIMINGINFO_H_
