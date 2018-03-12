// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCCATracking.cxx
/// \brief This task tests the TPC CA Tracking library
/// \author David Rohr

#define BOOST_TEST_MODULE Test TPC CATracking
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "TPCReconstruction/TPCCATracking.h"

#include <vector>
#include <iostream>
#include <iomanip>

using namespace o2::dataformats;

namespace o2
{
namespace TPC
{

/// @brief Test 1 basic class IO tests
BOOST_AUTO_TEST_CASE(CATracking_test1)
{
  TPCCATracking tracker;
  tracker.initialize("");
  std::vector<TrackTPC> tracks;
  std::vector<ClusterNativeContainer> cont(Constants::MAXGLOBALPADROW);

  for (int i = 0; i < Constants::MAXGLOBALPADROW; i++) {
    cont[i].sector = 0;
    cont[i].globalPadRow = i;
    cont[i].clusters.resize(1);
    cont[i].clusters[0].setTimeFlags(0, 0);
    cont[i].clusters[0].setPad(0);
    cont[i].clusters[0].setSigmaTime(1);
    cont[i].clusters[0].setSigmaPad(1);
    cont[i].clusters[0].qMax = 10;
    cont[i].clusters[0].qTot = 50;
  }
  std::unique_ptr<ClusterNativeAccessFullTPC> clusters =
    TPCClusterFormatHelper::accessNativeContainerArray(cont, nullptr);

  int retVal = tracker.runTracking(*clusters, &tracks, nullptr);
  BOOST_CHECK_EQUAL(retVal, 0);
  BOOST_CHECK_EQUAL((int)tracks.size(), 1);
}
}
}
