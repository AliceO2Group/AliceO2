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

/// \file testGPUGeometry.cxx
/// \brief Compare GPUTPCGeometry.h to o2::tpc::Mapper
/// \author David Rohr

#define BOOST_TEST_MODULE Test TPC GPUTPCGeometry
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataFormatsTPC/Constants.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/ParameterDetector.h"
#include "GPUTPCGeometry.h"

using namespace o2::gpu;

namespace o2
{
namespace tpc
{
/// @brief Test 1 basic class IO tests
BOOST_AUTO_TEST_CASE(GPUTPCGeometry_test1)
{
  const Mapper& mapper = Mapper::instance();
  const GPUTPCGeometry geo;
  const auto regions = mapper.getMapPadRegionInfo();

  BOOST_CHECK_EQUAL(mapper.getNumberOfPadRegions(), geo.NRegions());
  BOOST_CHECK_EQUAL(mapper.NSECTORS, geo.NSECTORS);
  BOOST_CHECK_EQUAL(mapper.PADROWS, geo.NROWS);

  auto& detParam = ParameterDetector::Instance();
  BOOST_CHECK_EQUAL(detParam.TPClength, geo.TPCLength());

  /*for (unsigned int i = 0; i < mapper.NPARTITIONS; i++) {
    BOOST_CHECK_EQUAL(???, geo.GetSectorFECOffset(i)); // TODO: Get value from mapper and compare!
  }*/

  for (unsigned int i = 0; i < mapper.getNumberOfPadRegions(); i++) {
    BOOST_CHECK_EQUAL(mapper.ROWSPERREGION[i], geo.GetRegionRows(i));
    BOOST_CHECK_EQUAL(mapper.ROWOFFSET[i], geo.GetRegionStart(i));
    // BOOST_CHECK_EQUAL(???, geo.GetSampaMapping(i)); // TODO: Get value from mapper and compare!
    // BOOST_CHECK_EQUAL(???, geo.GetChannelOffset(i)); // TODO: Get value from mapper and compare!
    BOOST_CHECK_EQUAL(regions[i].getPadHeight(), geo.PadHeightByRegion(i));
    BOOST_CHECK_EQUAL(regions[i].getPadWidth(), geo.PadWidthByRegion(i));
  }

  for (unsigned int i = 0; i < mapper.PADROWS; i++) {
    BOOST_CHECK_EQUAL(mapper.REGION[i], geo.GetRegion(i));
    unsigned int region = mapper.REGION[i];
    BOOST_CHECK_EQUAL(regions[region].getPadsInRowRegion(mapper.getLocalRowFromGlobalRow(i)), geo.NPads(i));
    const auto& pos = mapper.padCentre(mapper.getGlobalPadNumber(mapper.getLocalRowFromGlobalRow(i), 0, region));
    BOOST_CHECK_EQUAL(pos.x(), geo.Row2X(i));
  }
}
} // namespace tpc
} // namespace o2
