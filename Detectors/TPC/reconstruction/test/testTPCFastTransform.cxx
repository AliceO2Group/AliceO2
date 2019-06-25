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
#include "TPCReconstruction/TPCFastTransformHelperO2.h"
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

using namespace o2::gpu;

namespace o2
{
namespace tpc
{

/// @brief Test 1 basic class IO tests
BOOST_AUTO_TEST_CASE(FastTransform_test1)
{
  std::unique_ptr<TPCFastTransform> fastTransformPtr(TPCFastTransformHelperO2::instance()->create(0));

  TPCFastTransform& fastTransform = *fastTransformPtr;

  const Mapper& mapper = Mapper::instance();

  const TPCFastTransformGeo& geo = fastTransform.getGeometry();

  BOOST_CHECK_EQUAL(geo.test(), 0);

  BOOST_CHECK_EQUAL(geo.getNumberOfSlices(), Sector::MAXSECTOR);
  BOOST_CHECK_EQUAL(geo.getNumberOfRows(), mapper.getNumberOfRows());

  double maxDx = 0, maxDy = 0;

  for (int row = 0; row < geo.getNumberOfRows(); row++) {

    int nPads = geo.getRowInfo(row).maxPad + 1;

    BOOST_CHECK_EQUAL(nPads, mapper.getNumberOfPadsInRowSector(row));

    double x = geo.getRowInfo(row).x;

    // check if calculated pad positions are equal to the real ones

    for (int pad = 0; pad < nPads; pad++) {
      const GlobalPadNumber p = mapper.globalPadNumber(PadPos(row, pad));
      const PadCentre& c = mapper.padCentre(p);
      float u = 0, v = 0;
      fastTransform.convPadTimeToUV(0, row, pad, 0, u, v, 0.);

      double dx = x - c.X();
      double dy = u - (-c.Y()); // diferent sign convention for Y coordinate in the map
      BOOST_CHECK(fabs(dx) < 1.e-6);
      BOOST_CHECK(fabs(dy) < 1.e-5);
      if (fabs(dy) >= 1.e-5) {
        std::cout << "row " << row << " pad " << pad << " y calc " << u << " y in map " << -c.Y() << " dy " << dy << std::endl;
      }
      if (fabs(maxDx) < fabs(dx)) {
        maxDx = dx;
      }
      if (fabs(maxDy) < fabs(dy)) {
        maxDy = dy;
      }
    }
  }

  BOOST_CHECK(fabs(maxDx) < 1.e-6);
  BOOST_CHECK(fabs(maxDy) < 1.e-5);
}

BOOST_AUTO_TEST_CASE(FastTransform_test_setSpaceChargeCorrection)
{
  auto correctionFunction = [](const double XYZ[3], double dXdYdZ[3]) {
    dXdYdZ[0] = 1.;
    dXdYdZ[1] = 2.;
    dXdYdZ[2] = 3.;
  };

  TPCFastTransformHelperO2::instance()->setSpaceChargeCorrection(correctionFunction);

  std::unique_ptr<TPCFastTransform> fastTransform(TPCFastTransformHelperO2::instance()->create(0));

  const TPCFastTransformGeo& geo = fastTransform->getGeometry();

  double statDiff = 0., statN = 0.;

  for (int slice = 0; slice < geo.getNumberOfSlices(); slice += 1) {
    // std::cout << "slice " << slice << " ... " << std::endl;

    const TPCFastTransformGeo::SliceInfo& sliceInfo = geo.getSliceInfo(slice);

    for (int row = 0; row < geo.getNumberOfRows(); row++) {

      int nPads = geo.getRowInfo(row).maxPad + 1;

      for (int pad = 0; pad < nPads; pad += 10) {

        for (float time = 0; time < 1000; time += 30) {
          //std::cout<<"slice "<<slice<<" row "<<row<<" pad "<<pad<<" time "<<time<<std::endl;

          fastTransform->setApplyDistortionOff();
          float x0, y0, z0;
          fastTransform->Transform(slice, row, pad, time, x0, y0, z0);

          BOOST_CHECK_EQUAL(geo.test(slice, row, y0, z0), 0);

          fastTransform->setApplyDistortionOn();
          float x1, y1, z1;
          fastTransform->Transform(slice, row, pad, time, x1, y1, z1);

          // local 2 global
          float gx0, gy0, gz0;
          geo.convLocalToGlobal(slice, x0, y0, z0, gx0, gy0, gz0);
          float gx1, gy1, gz1;
          geo.convLocalToGlobal(slice, x1, y1, z1, gx1, gy1, gz1);

          double xyz[3] = { gx0, gy0, gz0 };
          double d[3] = { 0, 0, 0 };
          correctionFunction(xyz, d);
          statDiff += fabs((gx1 - gx0) - d[0]) + fabs((gy1 - gy0) - d[1]) + fabs((gz1 - gz0) - d[2]);
          statN += 3;
          //std::cout << (x1g-x0g) - d[0]<<" "<< (y1g-y0g) - d[1]<<" "<< (z1g-z0g) - d[2]<<std::endl;
        }
      }
    }
  }
  if (statN > 0)
    statDiff /= statN;
  //std::cout<<"average difference in distortion "<<statDiff<<" cm "<<std::endl;
  BOOST_CHECK_MESSAGE(fabs(statDiff) < 1.e-4, "test of distortion map failed, average difference " << statDiff << " cm is too large");
}

} // namespace tpc
} // namespace o2
