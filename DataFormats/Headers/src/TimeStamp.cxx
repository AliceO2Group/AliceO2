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

/// @file   TimeStamp.cxx
/// @author Matthias Richter
/// @since  2017-01-25
/// @brief  A std chrono implementation of LHC clock and timestamp

#include "Headers/TimeStamp.h"

using namespace o2::header;

// the only reason for the cxx file is the implementation of the
// constants
TimeStamp::TimeUnitID const TimeStamp::sClockLHC("AC");
TimeStamp::TimeUnitID const TimeStamp::sMicroSeconds("US");
