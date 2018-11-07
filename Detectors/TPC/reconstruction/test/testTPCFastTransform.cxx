// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testTPCFastTransform.cxx
/// \brief This task tests the TPC Fast Transformation
/// \author Sergey Gorbunov

#define BOOST_TEST_MODULE Test TPC Fast Transformation
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "TPCReconstruction/TPCFastTransformManagerO2.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCFastTransform.h"
#include "Riostream.h"
#include "FairLogger.h"

#include <vector>
#include <iostream>
#include <iomanip>

using namespace ali_tpc_common::tpc_fast_transformation;

namespace o2
{
namespace TPC
{

/// @brief Test 1 basic class IO tests
BOOST_AUTO_TEST_CASE(FastTransform_test1)
{
  TPCFastTransformManagerO2 manager;
  TPCFastTransform fastTransform;
  manager.create(fastTransform, 0);

  Mapper& mapper = Mapper::instance();

  BOOST_CHECK_EQUAL(fastTransform.getNumberOfSlices(), Sector::MAXSECTOR);
  BOOST_CHECK_EQUAL(fastTransform.getNumberOfRows(), mapper.getNumberOfRows());

  double maxDx = 0, maxDy = 0;

  for (int row = 0; row < fastTransform.getNumberOfRows(); row++) {

    int nPads = fastTransform.getRowInfo(row).maxPad + 1;

    BOOST_CHECK_EQUAL(nPads, mapper.getNumberOfPadsInRowSector(row));

    double x = fastTransform.getRowInfo(row).x;

    // check if calculated pad positions are equal to the real ones

    for (int pad = 0; pad < nPads; pad++) {
      const GlobalPadNumber p = mapper.globalPadNumber(PadPos(row, pad));
      const PadCentre& c = mapper.padCentre(p);
      float u = 0, v = 0;
      int err = fastTransform.convPadTimeToUV(0, row, pad, 0, u, v, 0.);
      BOOST_CHECK_EQUAL(err, 0);

      double dx = x - c.X();
      double dy = u - (-c.Y()); // diferent sign convention for Y coordinate in the map
      BOOST_CHECK(fabs(dx) < 1.e-6);
      BOOST_CHECK(fabs(dy) < 1.e-5);
      if (fabs(dy) >= 1.e-5) {
        std::cout << "row " << row << " pad " << pad << " y calc " << u << " y in map " << -c.Y() << " dy " << dy << std::endl;
      }
      if (fabs(maxDx) < fabs(dx))
        maxDx = dx;
      if (fabs(maxDy) < fabs(dy))
        maxDy = dy;
    }
  }

  BOOST_CHECK(fabs(maxDx) < 1.e-6);
  BOOST_CHECK(fabs(maxDy) < 1.e-5);
}
}
} // namespaces
