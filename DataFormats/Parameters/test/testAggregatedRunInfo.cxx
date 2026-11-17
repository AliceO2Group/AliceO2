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

#define BOOST_TEST_MODULE Test AggregatedRunInfo
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataFormatsParameters/AggregatedRunInfo.h"

using o2::parameters::AggregatedRunInfo;

// TF alignment shift for anchored MC (O2-7335)
BOOST_AUTO_TEST_CASE(TFAlignmentShift)
{
  // LHC26j17 run 535069: 90 s ramp-up shift of 1012094 orbits, 32 orbits per TF
  BOOST_CHECK_EQUAL(AggregatedRunInfo::tfAlignmentShift(68516798, 67504704, 32), 30);
  // LHC26h3 run 572150: 5 s ramp-up shift of 56228 orbits, 128 orbits per TF
  BOOST_CHECK_EQUAL(AggregatedRunInfo::tfAlignmentShift(114127812, 114071584, 128), 36);
  // MC-prod-2026-v14: shift of 56256 orbits, aligned for 32 orbits per TF, not for 128
  BOOST_CHECK_EQUAL(AggregatedRunInfo::tfAlignmentShift(114127840, 114071584, 32), 0);
  BOOST_CHECK_EQUAL(AggregatedRunInfo::tfAlignmentShift(114127840, 114071584, 128), 64);
  // first MC orbit before the data SOR: the shift stays in [0, orbitsPerTF)
  BOOST_CHECK_EQUAL(AggregatedRunInfo::tfAlignmentShift(100, 130, 32), 2);
  BOOST_CHECK_EQUAL(AggregatedRunInfo::tfAlignmentShift(100, 132, 32), 0);
  // invalid TF length
  BOOST_CHECK_EQUAL(AggregatedRunInfo::tfAlignmentShift(68516798, 67504704, 0), 0);
}
