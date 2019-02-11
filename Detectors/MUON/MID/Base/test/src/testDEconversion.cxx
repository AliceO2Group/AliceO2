// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE midBaseConversion
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
// Keep this separate or clang format will sort the include
// thus breaking compilation
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include "MIDBase/LegacyUtility.h"

namespace bdata = boost::unit_test::data;

namespace o2
{
namespace mid
{

std::vector<int> deIndexes = { 0, 3, 6, 8, 36, 44, 9, 12, 17, 45, 53 };
std::vector<int> detElemIds = { 1114, 1117, 1102, 1104, 1113, 1105, 1214, 1217, 1204, 1213, 1205 };

BOOST_DATA_TEST_CASE(MID_convertToLegacyDeId, bdata::make(deIndexes) ^ detElemIds, deIndex, detElemId)
{
  BOOST_TEST(LegacyUtility::convertToLegacyDeId(deIndex) == detElemId);
}

BOOST_DATA_TEST_CASE(MID_convertFromLegacyDeId, bdata::make(deIndexes) ^ detElemIds, deIndex, detElemId)
{
  BOOST_TEST(LegacyUtility::convertFromLegacyDeId(detElemId) == deIndex);
}

} // namespace mid
} // namespace o2
