// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCSyncPatternMonitor.cxx
/// \brief This task tests the SyncPatternMonitor module of the TPC GBT frame reader
/// \author Sebastian Klewin

#define BOOST_TEST_MODULE Test TPC SyncPatternMonitor
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCReconstruction/SyncPatternMonitor.h"

#include <vector>
#include <iostream>
#include <iomanip>

namespace o2
{
namespace tpc
{

struct result {
  int value;
  int position;
};

/// @brief Test 1 basic class IO tests
BOOST_AUTO_TEST_CASE(SyncPatternMonitor_test1)
{
  SyncPatternMonitor mon(0, 0);

  BOOST_CHECK_EQUAL(mon.getPatternA(), 0x15);
  BOOST_CHECK_EQUAL(mon.getPatternB(), 0x0A);
  BOOST_CHECK_EQUAL(mon.getSyncStart(), 2);
}

/// @brief Test 2 valid sequences
BOOST_AUTO_TEST_CASE(SyncPatternMonitor_test2)
{
  SyncPatternMonitor mon;

  std::vector<short> SYNC_PATTERN{
    /*mon.getPatternA(),mon.getPatternA(),*/ mon.getPatternB(), mon.getPatternB(),
    mon.getPatternA(), mon.getPatternA(), mon.getPatternB(), mon.getPatternB(),
    mon.getPatternA(), mon.getPatternA(), mon.getPatternB(), mon.getPatternB(),
    mon.getPatternA(), mon.getPatternA(), mon.getPatternB(), mon.getPatternB(),
    mon.getPatternA(), mon.getPatternA(), mon.getPatternA(), mon.getPatternA(),
    mon.getPatternB(), mon.getPatternB(), mon.getPatternB(), mon.getPatternB(),
    mon.getPatternA(), mon.getPatternA(), mon.getPatternA(), mon.getPatternA(),
    mon.getPatternB(), mon.getPatternB(), mon.getPatternB(), mon.getPatternB()};

  // loop over 4 possible positions
  for (int pos = 0; pos < 4; ++pos) {
    mon.reset();
    std::vector<short> test1_vec(4 + 4 + 2 + pos, mon.getPatternB());
    test1_vec.insert(test1_vec.begin() + 4 + 2 + pos, SYNC_PATTERN.begin(), SYNC_PATTERN.end());
    result res{pos, 4 + 32 + pos};

    for (int i = 0; i < test1_vec.size() - 4; i += 4) {
      if (res.position - i <= 4) {
        mon.addSequence(test1_vec[i], test1_vec[i + 1], test1_vec[i + 2], test1_vec[i + 3]);
        BOOST_CHECK_EQUAL(mon.getPosition(), res.value);
      } else {
        mon.addSequence(test1_vec[i], test1_vec[i + 1], test1_vec[i + 2], test1_vec[i + 3]);
        BOOST_CHECK_EQUAL(mon.getPosition(), -1);
      }
    }
  }
}

/// @brief Test 3 valid sequences, 1 pattern replaced with something else
BOOST_AUTO_TEST_CASE(SyncPatternMonitor_test3)
{
  SyncPatternMonitor mon;

  std::vector<short> SYNC_PATTERN{
    /*mon.getPatternA(),mon.getPatternA(),*/ mon.getPatternB(), mon.getPatternB(),
    mon.getPatternA(), mon.getPatternA(), mon.getPatternB(), mon.getPatternB(),
    mon.getPatternA(), mon.getPatternA(), mon.getPatternB(), mon.getPatternB(),
    mon.getPatternA(), mon.getPatternA(), mon.getPatternB(), mon.getPatternB(),
    mon.getPatternA(), mon.getPatternA(), mon.getPatternA(), mon.getPatternA(),
    mon.getPatternB(), mon.getPatternB(), mon.getPatternB(), mon.getPatternB(),
    mon.getPatternA(), mon.getPatternA(), mon.getPatternA(), mon.getPatternA(),
    mon.getPatternB(), mon.getPatternB(), mon.getPatternB(), mon.getPatternB()};

  // loop over 4 possible positions
  for (int pos = 0; pos < 4; ++pos) {
    std::vector<short> test1_vec(4 + 4 + 2 + pos, mon.getPatternB());
    test1_vec.insert(test1_vec.begin() + 4 + 2 + pos, SYNC_PATTERN.begin(), SYNC_PATTERN.end());
    result res{pos, 4 + 32 + pos};

    // loop over all positions of sync pattern in vector and replace with different pattern
    for (int v = 4 + 2 + pos; v < 4 + 2 + pos + SYNC_PATTERN.size(); ++v) {
      short old_Value = test1_vec[v];
      test1_vec[v] = 0x0;

      mon.reset();
      for (int i = 0; i < test1_vec.size() - 4; i += 4) {
        mon.addSequence(test1_vec[i], test1_vec[i + 1], test1_vec[i + 2], test1_vec[i + 3]);
        BOOST_CHECK_EQUAL(mon.getPosition(), -1);
      }
      test1_vec[v] = old_Value;
    }
  }
}

} // namespace tpc
} // namespace o2
