// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   testClusterHardware.cxx
/// @brief  Unit test for the TPC ClusterHardware data struct
/// \author David Rohr

#define BOOST_TEST_MODULE Test TPC DataFormats
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include "../include/DataFormatsTPC/ClusterHardware.h"
#include <boost/test/unit_test.hpp>
#include <iomanip>
#include <iostream>

namespace o2
{
namespace tpc
{
BOOST_AUTO_TEST_CASE(ClusterHardware_test1)
{
  ClusterHardware c;
  float pad = 0;
  float time = 0;
  float qTot = 0.0625;
  int qMax = 1;
  int flags = 0;
  int row = 0;
  float sigmaPad2 = 0;
  float sigmaTime2 = 0;
  do {
    c.setCluster(pad, time, sigmaPad2, sigmaTime2, qMax, qTot, row, flags);
    BOOST_CHECK_EQUAL(c.getFlags(), flags);
  } while (++flags < 256);
  do {
    c.setCluster(pad, time, sigmaPad2, sigmaTime2, qMax, qTot, row, flags);
    BOOST_CHECK_EQUAL(c.getRow(), row);
  } while (++row < 32);
  do {
    c.setCluster(pad, time, sigmaPad2, sigmaTime2, qMax, qTot, row, flags);
    BOOST_CHECK_EQUAL(c.getQTotFloat(), qTot);
  } while (++qTot <= 1023 * 25);
  do {
    c.setCluster(pad, time, sigmaPad2, sigmaTime2, qMax, qTot, row, flags);
    BOOST_CHECK_EQUAL(c.getQMax(), qMax);
  } while (++qMax < 1024);
  for (int i = 0; i < 3; i++) {
    qTot = i == 0 ? 0.25 : i == 1 ? (1023 * 10) : (1023 * 25 - 0.25);
    int maxSigma = i <= 1 ? 4 : 2;
    pad = time = 0;
    do {
      sigmaPad2 = sigmaTime2 = 0;
      c.setCluster(pad, time, sigmaPad2, sigmaTime2, qMax, qTot, row, flags);
      BOOST_CHECK_EQUAL(c.getPad(), pad);
      do {
        break;
        /*pad = 73.0;//828125;
        time = 0.0;//601562;
        sigmaPad2 = 0.458671;
        sigmaTime2 = 0.432741;
        qTot = 130.391449;
        qMax = 41.580238;*/
        c.setCluster(pad, time, sigmaPad2, sigmaTime2, qMax, qTot, row, flags);
        BOOST_CHECK_EQUAL(c.getSigmaPad2(), sigmaPad2);
      } while ((sigmaPad2 += 0.25) <= maxSigma);
    } while ((pad += 0.5) <= 255);
    do {
      break;
      sigmaPad2 = sigmaTime2 = 0;
      c.setCluster(pad, time, sigmaPad2, sigmaTime2, qMax, qTot, row, flags);
      BOOST_CHECK_EQUAL(c.getTimeLocal(), time);
      do {
        c.setCluster(pad, time, sigmaPad2, sigmaTime2, qMax, qTot, row, flags);
        BOOST_CHECK_EQUAL(c.getSigmaTime2(), sigmaTime2);
      } while ((sigmaTime2 += 0.25) <= maxSigma);
    } while ((time += 0.5) <= 511);
  }
}
} // namespace tpc
} // namespace o2
