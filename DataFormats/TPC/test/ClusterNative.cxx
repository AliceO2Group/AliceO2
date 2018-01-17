// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterNative.cxx
/// @since  2018-01-17
/// @brief  Unit test for the TPC ClusterNative data struct

#define BOOST_TEST_MODULE Test TPC DataFormats
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include "../include/DataFormatsTPC/ClusterNative.h"

namespace o2 {
namespace DataFormat {
namespace TPC {

// check function for different versions of the cluster type, versions
// can differ by internal members and scaling factors
template<typename ClusterType>
bool checkClusterType()
{
  constexpr float step = 0.01;
  constexpr auto padSeparation = 1./ClusterType::scalePadPacked;
  constexpr auto timeSeparation = 1./ClusterType::scaleTimePacked;
  constexpr auto sigmaPadSeparation = 1./ClusterType::scaleSigmaPadPacked;
  constexpr auto sigmaTimeSeparation = 1./ClusterType::scaleSigmaTimePacked;

  // the step size must be small then the achievable separation of values
  static_assert(step < padSeparation, "inconsistent step size");
  static_assert(step < timeSeparation, "inconsistent step size");

  ClusterType somecluster;
  for (float v = 0.; v < 5.; v += step) {
    somecluster.setPad(v);
    auto readback = somecluster.getPad();
    auto delta = padSeparation;
    BOOST_REQUIRE(readback > v - delta && readback < v + delta);

    v += step;
    somecluster.setTime(v);
    readback = somecluster.getTime();
    delta = timeSeparation;
    BOOST_REQUIRE(readback > v - delta && readback < v + delta);

    v += step;
    somecluster.setSigmaPad(v);
    readback = somecluster.getSigmaPad();
    delta = sigmaPadSeparation;
    BOOST_REQUIRE(readback > v - delta && readback < v + delta);

    v += step;
    somecluster.setSigmaTime(v);
    readback = somecluster.getSigmaTime();
    delta = sigmaTimeSeparation;
    BOOST_REQUIRE(readback > v - delta && readback < v + delta);

  }

  // currently a time frame is supposed to be 256 orbits at most, which is less
  // than 256 TPC drift lengths; 9 bit time sampling per drift length
  float maxTime = 256 * 512;
  uint8_t flags = 0x66;
  somecluster.setTimeFlags(maxTime, flags);
  auto read = somecluster.getTime();
  BOOST_REQUIRE(read > maxTime - timeSeparation && read < maxTime + timeSeparation);
  BOOST_REQUIRE(somecluster.getFlags() == 0x66);

  // check that there is no crosstalk to the flags field if time is too large
  somecluster.setFlags(0);
  somecluster.setTime(4 * maxTime);
  BOOST_REQUIRE(somecluster.getFlags() == 0);

  return true;
}

BOOST_AUTO_TEST_CASE(test_tpc_clusternative)
{
  checkClusterType<ClusterNative>();
}

} // namespace TPC
} // namespace DataFormat
} // namespace o2
