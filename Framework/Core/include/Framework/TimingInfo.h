// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_TIMINGINFO_H
#define FRAMEWORK_TIMINGINFO_H

#include <cstddef>

/// This class holds the information about timing
/// of the messages being processed.
struct TimingInfo {
  size_t timeslice; /// the timeslice associated to current processing
};

#endif // Timing information for the current computation
