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

#define BOOST_TEST_MODULE error test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "MCHBase/Error.h"
#include <map>

using o2::mch::Error;
using o2::mch::ErrorGroup;
using o2::mch::ErrorType;

/// @brief expected ID of each error group
// this needs to be updated every time a new group is added
// changing the IDs of existing groups will break the backward compatibility
const std::map<ErrorGroup, uint8_t> errorGroupIds = {
  {ErrorGroup::Unassigned, 0U},
  {ErrorGroup::Decoding, 1U},
  {ErrorGroup::Filtering, 2U},
  {ErrorGroup::TimeClustering, 3U},
  {ErrorGroup::PreClustering, 4U},
  {ErrorGroup::Clustering, 5U},
  {ErrorGroup::Tracking, 6U}};

/// @brief expected ID of each error type
// this needs to be updated every time a new type is added
// changing the IDs of existing types will break the backward compatibility
const std::map<ErrorType, uint32_t> errorTypeIds = {
  {ErrorType::PreClustering_MultipleDigitsInSamePad, 67108864U},
  {ErrorType::PreClustering_LostDigit, 67108865U},
  {ErrorType::Clustering_TooManyLocalMaxima, 83886080U},
  {ErrorType::Tracking_TooManyCandidates, 100663296U},
  {ErrorType::Tracking_TooLong, 100663297U}};

BOOST_AUTO_TEST_CASE(ErrorGroupConsistency)
{
  BOOST_CHECK_EQUAL(errorGroupIds.size(), Error::groupNames.size());
  for (auto [group, id] : errorGroupIds) {
    BOOST_CHECK_EQUAL(static_cast<uint8_t>(group), id);
    BOOST_CHECK_EQUAL(Error::groupNames.count(group), 1);
  }
}

BOOST_AUTO_TEST_CASE(ErrorTypeConsistency)
{
  BOOST_CHECK_EQUAL(errorTypeIds.size(), Error::typeNames.size());
  BOOST_CHECK_EQUAL(errorTypeIds.size(), Error::typeDescriptions.size());
  for (auto [type, id] : errorTypeIds) {
    BOOST_CHECK_EQUAL(static_cast<uint32_t>(type), id);
    BOOST_CHECK_EQUAL(Error::typeNames.count(type), 1);
    BOOST_CHECK_EQUAL(Error::typeDescriptions.count(type), 1);
    BOOST_CHECK_EQUAL(errorGroupIds.count(o2::mch::errorGroup(type)), 1);
  }
}
