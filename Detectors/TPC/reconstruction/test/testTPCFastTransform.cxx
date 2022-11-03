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
#include <fairlogger/Logger.h>

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

  // init some transformation w/o space charge correction
  // to initialize TPCFastTransformGeo geometry

  std::unique_ptr<TPCFastTransform> fastTransform0(TPCFastTransformHelperO2::instance()->create(0));
  const TPCFastTransformGeo& geo = fastTransform0->getGeometry();

  auto correctionUV = [&](int roc, int /*row*/, const double u, const double v, double& dX, double& dU, double& dV) {
    // float lx = geo.getRowInfo(row).x;
    dX = 1. + 1 * u + 0.1 * u * u;
    dU = 2. + 0.2 * u + 0.002 * u * u; // + 0.001 * u * u * u;
    dV = 3. + 0.1 * v + 0.01 * v * v;  //+ 0.0001 * v * v * v;
  };

  auto correctionLocal = [&](int roc, int row, double ly, double lz,
                             double& dx, double& dly, double& dlz) {
    float u, v;
    geo.convLocalToUV(roc, ly, lz, u, v);
    double du, dv;
    correctionUV(roc, row, u, v, dx, du, dv);
    float ly1, lz1;
    geo.convUVtoLocal(roc, u + du, v + dv, ly1, lz1);
    dly = ly1 - ly;
    dlz = lz1 - lz;
  };

  int nRocs = geo.getNumberOfSlices();
  int nRows = geo.getNumberOfRows();
  TPCFastSpaceChargeCorrectionMap& scData = TPCFastTransformHelperO2::instance()->getCorrectionMap();
  scData.init(nRocs, nRows);

  for (int iRoc = 0; iRoc < nRocs; iRoc++) {
    for (int iRow = 0; iRow < nRows; iRow++) {
      double dsu = 1. / (3 * 8 - 3);
      double dsv = 1. / (3 * 20 - 3);
      for (double su = 0.f; su < 1.f + .5 * dsu; su += dsv) {
        for (double sv = 0.f; sv < 1.f + .5 * dsv; sv += dsv) {
          float ly = 0.f, lz = 0.f;
          geo.convScaledUVtoLocal(iRoc, iRow, su, sv, ly, lz);
          double dx, dy, dz;
          correctionLocal(iRoc, iRow, ly, lz, dx, dy, dz);
          scData.addCorrectionPoint(iRoc, iRow,
                                    ly, lz, dx, dy, dz);
        }
      }
    } // row
  }   // slice

  std::unique_ptr<TPCFastTransform> fastTransform(TPCFastTransformHelperO2::instance()->create(0));

  int err = fastTransform->writeToFile("tmpTestTPCFastTransform.root");

  BOOST_CHECK_EQUAL(err, 0);

  TPCFastTransform* fromFile = TPCFastTransform::loadFromFile("tmpTestTPCFastTransform.root");

  BOOST_CHECK(fromFile != nullptr);

  double statDiff = 0., statN = 0.;
  double statDiffFile = 0., statNFile = 0.;

  for (int slice = 0; slice < geo.getNumberOfSlices(); slice += 1) {
    //std::cout << "slice " << slice << " ... " << std::endl;

    const TPCFastTransformGeo::SliceInfo& sliceInfo = geo.getSliceInfo(slice);

    float lastTimeBin = fastTransform->getMaxDriftTime(slice, 0.f);

    for (int row = 0; row < geo.getNumberOfRows(); row++) {

      int nPads = geo.getRowInfo(row).maxPad + 1;

      for (int pad = 0; pad < nPads; pad += 10) {

        for (float time = 0; time < lastTimeBin; time += 30) {
          //std::cout<<"slice "<<slice<<" row "<<row<<" pad "<<pad<<" time "<<time<<std::endl;

          fastTransform->setApplyCorrectionOff();
          float x0, y0, z0;
          fastTransform->Transform(slice, row, pad, time, x0, y0, z0);

          BOOST_CHECK_EQUAL(geo.test(slice, row, y0, z0), 0);

          fastTransform->setApplyCorrectionOn();
          float x1, y1, z1;
          fastTransform->Transform(slice, row, pad, time, x1, y1, z1);

          // local to UV
          float u0, v0, u1, v1;
          geo.convLocalToUV(slice, y0, z0, u0, v0);
          geo.convLocalToUV(slice, y1, z1, u1, v1);
          double dx, du, dv;
          correctionUV(slice, row, u0, v0, dx, du, dv);
          statDiff += fabs((x1 - x0) - dx) + fabs((u1 - u0) - du) + fabs((v1 - v0) - dv);
          statN += 3;
          //std::cout << (x1 - x0) - dx << " " << (u1 - u0) - du << " " << (v1 - v0) - dv << std::endl; //": v0 " << v0 <<" z0 "<<z0<<" v1 "<< v1<<" z1 "<<z1 << std::endl;
          //BOOST_CHECK_MESSAGE(0, "SG");

          float x1f, y1f, z1f;
          fromFile->Transform(slice, row, pad, time, x1f, y1f, z1f);
          statDiffFile += fabs(x1f - x1) + fabs(y1f - y1) + fabs(z1f - z1);
          statNFile += 3;
        }
      }
    }
  }
  if (statN > 0) {
    statDiff /= statN;
  }

  if (statNFile > 0) {
    statDiffFile /= statNFile;
  }

  std::cout << "average difference in correction " << statDiff << " cm " << std::endl;
  BOOST_CHECK_MESSAGE(fabs(statDiff) < 1.e-3, "test of correction map failed, average difference " << statDiff << " cm is too large");
  BOOST_CHECK_MESSAGE(fabs(statDiffFile) < 1.e-10, "test of file streamer failed, average difference " << statDiffFile << " cm is too large");

  double maxDeviation = fastTransform->getCorrection().testInverse();
  std::cout << "max deviation for inverse correction " << maxDeviation << " cm " << std::endl;
  BOOST_CHECK_MESSAGE(fabs(maxDeviation) < 1.e-2, "test of inverse correction map failed, max difference " << maxDeviation << " cm is too large");
}

} // namespace tpc
} // namespace o2
