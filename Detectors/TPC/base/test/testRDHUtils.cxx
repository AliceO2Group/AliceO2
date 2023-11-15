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

#define BOOST_TEST_MODULE Test TPC CalDet class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <vector>

#include "Headers/RAWDataHeader.h"
#include "TPCBase/RDHUtils.h"

namespace o2
{
namespace tpc
{
using namespace rdh_utils;

struct FEEDetails {
  FEEDetails(FEEIDType c, FEEIDType e, FEEIDType l) : cru(c), endpoint(e), link(l), feeID((cru << 7) | (endpoint << 6) | link), userLogic(l == UserLogicLinkID) {}
  FEEIDType cru{};
  FEEIDType endpoint{};
  FEEIDType link{};
  FEEIDType feeID{};
  bool userLogic{};
};

BOOST_AUTO_TEST_CASE(testRDHUtils)
{
  // coose some random values
  std::vector<FEEDetails> testDetails;
  testDetails.emplace_back(359, 1, 15);
  testDetails.emplace_back(1, 0, 10);
  testDetails.emplace_back(100, 1, 8);
  testDetails.emplace_back(142, 0, 1);
  testDetails.emplace_back(225, 1, 9);

  o2::header::RAWDataHeaderV4 rdh4;
  o2::header::RAWDataHeaderV5 rdh5;
  o2::header::RAWDataHeaderV6 rdh6;
  o2::header::RAWDataHeaderV7 rdh7;

  for (const auto& fee : testDetails) {
    const auto feeID = getFEEID(fee.cru, fee.endpoint, fee.link);

    // default checks
    auto cru = getCRU(feeID);
    auto link = getLink(feeID);
    auto endpoint = getEndPoint(feeID);
    bool userLogic = isFromUserLogic(feeID);

    BOOST_CHECK_EQUAL(feeID, fee.feeID);
    BOOST_CHECK_EQUAL(cru, fee.cru);
    BOOST_CHECK_EQUAL(endpoint, fee.endpoint);
    BOOST_CHECK_EQUAL(link, fee.link);
    BOOST_CHECK_EQUAL(userLogic, fee.userLogic);

    // RDH v4 checks
    rdh4.feeId = feeID;
    cru = getCRU(rdh4);
    link = getLink(rdh4);
    endpoint = getEndPoint(rdh4);
    userLogic = isFromUserLogic(rdh4);

    BOOST_CHECK_EQUAL(feeID, fee.feeID);
    BOOST_CHECK_EQUAL(cru, fee.cru);
    BOOST_CHECK_EQUAL(endpoint, fee.endpoint);
    BOOST_CHECK_EQUAL(link, fee.link);
    BOOST_CHECK_EQUAL(userLogic, fee.userLogic);

    // RDH v5 checks
    rdh5.feeId = feeID;
    cru = getCRU(rdh5);
    link = getLink(rdh5);
    endpoint = getEndPoint(rdh5);
    userLogic = isFromUserLogic(rdh5);

    BOOST_CHECK_EQUAL(feeID, fee.feeID);
    BOOST_CHECK_EQUAL(cru, fee.cru);
    BOOST_CHECK_EQUAL(endpoint, fee.endpoint);
    BOOST_CHECK_EQUAL(link, fee.link);
    BOOST_CHECK_EQUAL(userLogic, fee.userLogic);

    // RDH v6 checks
    rdh6.feeId = feeID;
    cru = getCRU(rdh6);
    link = getLink(rdh6);
    endpoint = getEndPoint(rdh6);
    userLogic = isFromUserLogic(rdh6);

    BOOST_CHECK_EQUAL(feeID, fee.feeID);
    BOOST_CHECK_EQUAL(cru, fee.cru);
    BOOST_CHECK_EQUAL(endpoint, fee.endpoint);
    BOOST_CHECK_EQUAL(link, fee.link);
    BOOST_CHECK_EQUAL(userLogic, fee.userLogic);

    // RDH v7 checks
    rdh7.feeId = feeID;
    cru = getCRU(rdh7);
    link = getLink(rdh7);
    endpoint = getEndPoint(rdh7);
    userLogic = isFromUserLogic(rdh7);

    BOOST_CHECK_EQUAL(feeID, fee.feeID);
    BOOST_CHECK_EQUAL(cru, fee.cru);
    BOOST_CHECK_EQUAL(endpoint, fee.endpoint);
    BOOST_CHECK_EQUAL(link, fee.link);
    BOOST_CHECK_EQUAL(userLogic, fee.userLogic);
  }
}

} // namespace tpc
} // namespace o2
