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

#define BOOST_TEST_MODULE trackable test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "MCHBase/Trackable.h"

BOOST_AUTO_TEST_CASE(MissingRequiredStationMeansNotTrackable)
{
  std::array<int, 10> items = {0, 0, 3, 4, 5, 6, 7, 8, 9, 10};
  std::array<bool, 5> requestStations = {true, true, true, true, true};
  auto moreCandidates = false;

  BOOST_CHECK_EQUAL(o2::mch::isTrackable(items, requestStations, moreCandidates),
                    false);
}

BOOST_AUTO_TEST_CASE(MissingNotRequiredStationIsOK)
{
  std::array<int, 10> items = {0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
  std::array<bool, 5> requestStations = {false, true, true, true, true};
  auto moreCandidates = false;

  BOOST_CHECK_EQUAL(o2::mch::isTrackable(items, requestStations, moreCandidates),
                    true);
}

BOOST_AUTO_TEST_CASE(WithoutMoreCandidatesOptionOnly2ItemsInSt45IsNotOK)
{
  std::array<int, 10> items = {1, 1, 1, 1, 1, 1, 0, 1, 0, 1};
  std::array<bool, 5> requestStations = {true, true, true, true, true};
  auto moreCandidates = false;

  BOOST_CHECK_EQUAL(o2::mch::isTrackable(items, requestStations, moreCandidates),
                    false);
}

BOOST_AUTO_TEST_CASE(WithMoreCandidatesOptionOnly2ItemsInSt45IsOK)
{
  std::array<int, 10> items = {1, 1, 1, 1, 1, 1, 0, 1, 0, 1};
  std::array<bool, 5> requestStations = {true, true, true, true, true};
  auto moreCandidates = true;

  BOOST_CHECK_EQUAL(o2::mch::isTrackable(items, requestStations, moreCandidates),
                    true);
}
