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

#define BOOST_TEST_MODULE Test MCHRaw NofPadsPerFeeId
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MCHMappingInterface/Segmentation.h"
#include "MCHRawElecMap/Mapper.h"
#include "MCHRawElecMap/FeeLinkId.h"
#include <iostream>
#include <vector>

using namespace o2::mch::raw;

int nofPadsPerFeeId(uint16_t feeid)
{
  int npads{0};
  auto feeLink2Solar = createFeeLink2SolarMapper<ElectronicMapperGenerated>();
  auto solars = getSolarUIDsPerFeeId<ElectronicMapperGenerated>(feeid);
  for (auto solar : solars) {
    auto dualSampas = getDualSampas<ElectronicMapperGenerated>(solar);
    for (auto ds : dualSampas) {
      const o2::mch::mapping::Segmentation& seg = o2::mch::mapping::segmentation(ds.deId());
      for (auto channel = 0; channel < 64; channel++) {
        auto paduid = seg.findPadByFEE(ds.dsId(), channel);
        if (seg.isValid(paduid)) {
          npads++;
        }
      }
    }
  }
  return npads;
}

BOOST_AUTO_TEST_CASE(NumberOfPadsPerFeeId)
{
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(0), 28672);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(1), 28672);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(2), 28672);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(3), 28672);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(4), 28672);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(5), 28672);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(6), 28672);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(7), 28672);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(8), 27894);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(9), 27852);

  BOOST_CHECK_EQUAL(nofPadsPerFeeId(10), 0);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(11), 0);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(12), 27894);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(13), 27852);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(14), 27894);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(15), 27852);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(16), 27894);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(17), 27852);

  BOOST_CHECK_EQUAL(nofPadsPerFeeId(18), 16800);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(19), 13024);

  BOOST_CHECK_EQUAL(nofPadsPerFeeId(20), 8752);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(21), 16800);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(22), 13024);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(23), 8752);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(24), 14400);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(25), 7808);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(26), 17088);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(27), 14400);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(28), 7808);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(29), 17088);

  BOOST_CHECK_EQUAL(nofPadsPerFeeId(30), 5984);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(31), 18400);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(32), 15856);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(33), 13968);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(34), 5984);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(35), 18400);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(36), 5984);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(37), 18400);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(38), 15856);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(39), 13968);

  BOOST_CHECK_EQUAL(nofPadsPerFeeId(40), 5984);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(41), 18400);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(42), 15856);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(43), 13968);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(44), 11184);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(45), 18576);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(46), 18912);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(47), 10976);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(48), 15856);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(49), 13968);

  BOOST_CHECK_EQUAL(nofPadsPerFeeId(50), 11184);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(51), 18576);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(52), 18912);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(53), 10976);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(54), 11184);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(55), 18576);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(56), 18912);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(57), 10976);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(58), 0);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(59), 0);

  BOOST_CHECK_EQUAL(nofPadsPerFeeId(60), 11184);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(61), 18576);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(62), 18912);
  BOOST_CHECK_EQUAL(nofPadsPerFeeId(63), 10976);

  BOOST_CHECK_EQUAL(nofPadsPerFeeId(64), 0);
}

BOOST_AUTO_TEST_CASE(TotalNumberOfPads)
{
  int npads{0};
  for (auto feeid = 0; feeid < 64; feeid++) {
    npads += nofPadsPerFeeId(feeid);
  }
#ifdef MCH_MAPPING_RUN3_AND_ABOVE
  BOOST_CHECK_EQUAL(npads, 1063528);
#else
  BOOST_CHECK_EQUAL(npads, 1064008);
#endif
}
