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

/// \file TPCFastTransformHelperO2.cxx
/// \author Sergey Gorbunov

#include "TPCReconstruction/TPCFastTransformHelperO2.h"

#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCFastTransform.h"
#include "Spline2DHelper.h"
#include "Riostream.h"
#include <fairlogger/Logger.h>

using namespace o2::gpu;

namespace o2
{
namespace tpc
{

TPCFastTransformHelperO2* TPCFastTransformHelperO2::sInstance = nullptr;

TPCFastTransformHelperO2* TPCFastTransformHelperO2::instance()
{
  // returns TPCFastTransformHelperO2 instance (singleton)
  if (!sInstance) {
    sInstance = new TPCFastTransformHelperO2();
    sInstance->init();
  }
  return sInstance;
}

void TPCFastTransformHelperO2::init()
{
  // initialize geometry

  const Mapper& mapper = Mapper::instance();

  const int nRows = mapper.getNumberOfRows();

  mGeo.startConstruction(nRows);

  auto& detParam = ParameterDetector::Instance();
  float tpcZlengthSideA = detParam.TPClength;
  float tpcZlengthSideC = detParam.TPClength;

  mGeo.setTPCzLength(tpcZlengthSideA, tpcZlengthSideC);

  mGeo.setTPCalignmentZ(0.);

  for (int iRow = 0; iRow < mGeo.getNumberOfRows(); iRow++) {
    Sector sector = 0;
    int regionNumber = 0;
    while (iRow >= mapper.getGlobalRowOffsetRegion(regionNumber) + mapper.getNumberOfRowsRegion(regionNumber)) {
      regionNumber++;
    }

    const PadRegionInfo& region = mapper.getPadRegionInfo(regionNumber);

    int nPads = mapper.getNumberOfPadsInRowSector(iRow);
    float padWidth = region.getPadWidth();

    const GlobalPadNumber pad = mapper.globalPadNumber(PadPos(iRow, nPads / 2));
    const PadCentre& padCentre = mapper.padCentre(pad);
    float xRow = padCentre.X();

    mGeo.setTPCrow(iRow, xRow, nPads, padWidth);
  }

  mGeo.finishConstruction();

  // check if calculated pad geometry is consistent with the map
  testGeometry(mGeo);

  mIsInitialized = 1;
}

std::unique_ptr<TPCFastTransform> TPCFastTransformHelperO2::create(Long_t TimeStamp, const int nKnotsY, const int nKnotsZ)
{
  /// initializes TPCFastTransform object

  // init geometry

  if (!mIsInitialized) {
    init();
  }

  TPCFastSpaceChargeCorrection correction;

  { // create the correction map

    const int nRows = mGeo.getNumberOfRows();
    const int nCorrectionScenarios = nRows / 10 + 1;

    correction.startConstruction(mGeo, nCorrectionScenarios);

    // init rows
    for (int row = 0; row < mGeo.getNumberOfRows(); row++) {
      int scenario = row / 10;
      if (scenario >= nCorrectionScenarios) {
        scenario = nCorrectionScenarios - 1;
      }
      correction.setRowScenarioID(row, scenario);
    }

    // adjust the number of knots and the knot positions for the TPC correction splines

    /*
     TODO: update the calibrator
    IrregularSpline2D3DCalibrator calibrator;
    calibrator.setRasterSize(41, 41);
    calibrator.setMaxNKnots(21, 21);
    calibrator.setMaximalDeviation(0.01);

    IrregularSpline2D3D raster;
    raster.constructRegular(101, 101);
    std::vector<float> rasterData(3 * raster.getNumberOfKnots());
    */
    for (int scenario = 0; scenario < nCorrectionScenarios; scenario++) {
      int row = scenario * 10;
      TPCFastSpaceChargeCorrection::SplineType spline;
      if (!mCorrectionMap.isInitialized() || row >= nRows) {
        spline.recreate(nKnotsY, nKnotsZ);
      } else {
        // TODO: update the calibrator
        spline.recreate(nKnotsY, nKnotsZ);
        /*
        // create the input function
        for (int knot = 0; knot < raster.getNumberOfKnots(); knot++) {
          float su = 0.f, sv = 0.f;
          raster.getKnotUV(knot, su, sv);
          float dx = 0.f, du = 0.f, dv = 0.f;
          const int slice = 0;
          getSpaceChargeCorrection(slice, row, su, sv, dx, du, dv);
          rasterData[3 * knot + 0] = dx;
          rasterData[3 * knot + 1] = du;
          rasterData[3 * knot + 2] = dv;
        }
        raster.correctEdges(rasterData.data());
        std::function<void(float, float, float&, float&, float&)> f;
        f = [&raster, &rasterData](float su, float sv, float& dx, float& du, float& dv) {
          raster.getSpline(rasterData.data(), su, sv, dx, du, dv);
        };

        calibrator.calibrateSpline(spline, f);
        std::cout << "calibrated spline for scenario " << scenario << ", TPC row " << row << ": knots u "
                  << spline.getGridU().getNumberOfKnots() << ", v "
                  << spline.getGridV().getNumberOfKnots() << std::endl;
                  */
      }
      correction.setSplineScenario(scenario, spline);
    }
    correction.finishConstruction();
  } // .. create the correction map

  std::unique_ptr<TPCFastTransform> fastTransformPtr(new TPCFastTransform);

  TPCFastTransform& fastTransform = *fastTransformPtr;

  { // create the fast transform object

    fastTransform.startConstruction(correction);

    // tell the transformation to apply the space charge corrections
    fastTransform.setApplyCorrectionOn();

    // set some initial calibration values, will be reinitialised later int updateCalibration()
    const float t0 = 0.;
    const float vDrift = 0.f;
    const float vdCorrY = 0.;
    const float ldCorr = 0.;
    const float tofCorr = 0.;
    const float primVtxZ = 0.;
    const long int initTimeStamp = -1;
    fastTransform.setCalibration(initTimeStamp, t0, vDrift, vdCorrY, ldCorr, tofCorr, primVtxZ);

    fastTransform.finishConstruction();
  }

  fillSpaceChargeCorrectionFromMap(fastTransform.getCorrection());
  updateCalibration(fastTransform, TimeStamp);

  return std::move(fastTransformPtr);
}

void TPCFastTransformHelperO2::fillSpaceChargeCorrectionFromMap(TPCFastSpaceChargeCorrection& correction)
{
  // now calculate correction map: dx,du,dv = ( origTransform() -> x,u,v) - fastTransformNominal:x,u,v
  // for the future: switch TOF correction off for a while

  if (mCorrectionMap.isInitialized()) {
    for (int slice = 0; slice < correction.getGeometry().getNumberOfSlices(); slice++) {
      for (int row = 0; row < correction.getGeometry().getNumberOfRows(); row++) {
        TPCFastSpaceChargeCorrection::SplineType& spline = correction.getSpline(slice, row);
        Spline2DHelper<float> helper;
        float* splineParameters = correction.getSplineData(slice, row);
        /* old style

        helper.setSpline(spline, 3, 3);
        auto F = [&](double su, double sv, double dxuv[3]) {
          getSpaceChargeCorrection(slice, row, su, sv, dxuv[0], dxuv[1], dxuv[2]);
        };
        helper.approximateFunction(splineParameters, 0., 1., 0., 1., F);
        */
        const std::vector<o2::gpu::TPCFastSpaceChargeCorrectionMap::CorrectionPoint>& data = mCorrectionMap.getPoints(slice, row);
        int nDataPoints = data.size();
        if (nDataPoints >= 4) {
          std::vector<double> pointSU(nDataPoints);
          std::vector<double> pointSV(nDataPoints);
          std::vector<double> pointCorr(3 * nDataPoints); // 3 dimensions
          for (int i = 0; i < nDataPoints; ++i) {
            double su, sv, dx, du, dv;
            getSpaceChargeCorrection(slice, row, data[i], su, sv, dx, du, dv);
            pointSU[i] = su;
            pointSV[i] = sv;
            pointCorr[3 * i + 0] = dx;
            pointCorr[3 * i + 1] = du;
            pointCorr[3 * i + 2] = dv;
          }
          helper.approximateDataPoints(spline, splineParameters, 0., 1., 0., 1., &pointSU[0],
                                       &pointSV[0], &pointCorr[0], nDataPoints);
        } else {
          for (int i = 0; i < spline.getNumberOfParameters(); i++) {
            splineParameters[i] = 0.f;
          }
        }
      } // row
    }   // slice
    correction.initInverse();
  } else {
    correction.setNoCorrection();
  }
}

int TPCFastTransformHelperO2::updateCalibration(TPCFastTransform& fastTransform, Long_t TimeStamp, float vDriftFactor, float vDriftRef, float driftTimeOffset)
{
  // Update the calibration with the new time stamp
  LOGP(debug, "Updating calibration: timestamp:{} vdriftFactor:{} vdriftRef:{}", TimeStamp, vDriftFactor, vDriftRef);
  if (!mIsInitialized) {
    init();
  }

  if (TimeStamp < 0) {
    return 0;
  }

  // search for the calibration database ...

  auto& detParam = ParameterDetector::Instance();
  auto& gasParam = ParameterGas::Instance();
  auto& elParam = ParameterElectronics::Instance();
  // start the initialization

  fastTransform.setTimeStamp(TimeStamp);
  if (vDriftRef == 0) {
    vDriftRef = ParameterGas::Instance().DriftV;
  }
  const double vDrift = elParam.ZbinWidth * vDriftRef * vDriftFactor; // cm/timebin

  // fast transform formula:
  // L = (t-t0)*(mVdrift + mVdriftCorrY*yLab ) + mLdriftCorr
  // Z = Z(L) +  tpcAlignmentZ
  // spline corrections for xyz
  // Time-of-flight correction: ldrift += dist-to-vtx*tofCorr

  const double t0 = (driftTimeOffset + elParam.getAverageShapingTime()) / elParam.ZbinWidth;

  const double vdCorrY = 0.;
  const double ldCorr = 0.;
  const double tofCorr = 0.;
  const double primVtxZ = 0.;

  fastTransform.setCalibration(TimeStamp, t0, vDrift, vdCorrY, ldCorr, tofCorr, primVtxZ);

  // The next line should not be needed
  // fastTransform.getCorrection().initInverse();
  // for the future: set back the time-of-flight correction

  return 0;
}

void TPCFastTransformHelperO2::getSpaceChargeCorrection(int slice, int row, o2::gpu::TPCFastSpaceChargeCorrectionMap::CorrectionPoint p,
                                                        double& su, double& sv, double& dx, double& du, double& dv)
{
  // get space charge correction in internal TPCFastTransform coordinates su,sv->dx,du,dv

  if (!mIsInitialized) {
    init();
  }

  // not corrected coordinates in u,v
  float u = 0.f, v = 0.f, fsu = 0.f, fsv = 0.f;
  mGeo.convLocalToUV(slice, p.mY, p.mZ, u, v);
  mGeo.convUVtoScaledUV(slice, row, u, v, fsu, fsv);
  su = fsu;
  sv = fsv;
  // corrected coordinates in u,v
  float u1 = 0.f, v1 = 0.f;
  mGeo.convLocalToUV(slice, p.mY + p.mDy, p.mZ + p.mDz, u1, v1);

  dx = p.mDx;
  du = u1 - u;
  dv = v1 - v;
}

void TPCFastTransformHelperO2::setGlobalSpaceChargeCorrection(
  std::function<void(int roc, double gx, double gy, double gz,
                     double& dgx, double& dgy, double& dgz)>
    correctionGlobal)
{
  auto correctionLocal = [&](int roc, int irow, double ly, double lz,
                             double& dlx, double& dly, double& dlz) {
    double lx = mGeo.getRowInfo(irow).x;
    float gx, gy, gz;
    mGeo.convLocalToGlobal(roc, lx, ly, lz, gx, gy, gz);
    double dgx, dgy, dgz;
    correctionGlobal(roc, gx, gy, gz, dgx, dgy, dgz);
    float lx1, ly1, lz1;
    mGeo.convGlobalToLocal(roc, gx + dgx, gy + dgy, gz + dgz, lx1, ly1, lz1);
    dlx = lx1 - lx;
    dly = ly1 - ly;
    dlz = lz1 - lz;
  };
  setLocalSpaceChargeCorrection(correctionLocal);
}

void TPCFastTransformHelperO2::setLocalSpaceChargeCorrection(
  std::function<void(int roc, int irow, double y, double z,
                     double& dx, double& dy, double& dz)>
    correctionLocal)
{
  /// set space charge correction in the local coordinates
  /// as a continious function

  int nRocs = mGeo.getNumberOfSlices();
  int nRows = mGeo.getNumberOfRows();
  mCorrectionMap.init(nRocs, nRows);

  for (int iRoc = 0; iRoc < nRocs; iRoc++) {
    for (int iRow = 0; iRow < nRows; iRow++) {
      double dsu = 1. / (3 * 8 - 3);
      double dsv = 1. / (3 * 20 - 3);
      for (double su = 0.f; su < 1.f + .5 * dsu; su += dsv) {
        for (double sv = 0.f; sv < 1.f + .5 * dsv; sv += dsv) {
          float ly, lz;
          mGeo.convScaledUVtoLocal(iRoc, iRow, su, sv, ly, lz);
          double dx, dy, dz;
          correctionLocal(iRoc, iRow, ly, lz, dx, dy, dz);
          mCorrectionMap.addCorrectionPoint(iRoc, iRow,
                                            ly, lz, dx, dy, dz);
        }
      }
    } // row
  }   // roc
}

void TPCFastTransformHelperO2::testGeometry(const TPCFastTransformGeo& geo) const
{
  const Mapper& mapper = Mapper::instance();

  if (geo.getNumberOfSlices() != Sector::MAXSECTOR) {
    LOG(fatal) << "Wrong number of sectors :" << geo.getNumberOfSlices() << " instead of " << Sector::MAXSECTOR << std::endl;
  }

  if (geo.getNumberOfRows() != mapper.getNumberOfRows()) {
    LOG(fatal) << "Wrong number of rows :" << geo.getNumberOfRows() << " instead of " << mapper.getNumberOfRows() << std::endl;
  }

  double maxDx = 0, maxDy = 0;

  for (int row = 0; row < geo.getNumberOfRows(); row++) {

    const int nPads = geo.getRowInfo(row).maxPad + 1;

    if (nPads != mapper.getNumberOfPadsInRowSector(row)) {
      LOG(fatal) << "Wrong number of pads :" << nPads << " instead of " << mapper.getNumberOfPadsInRowSector(row) << std::endl;
    }

    const double x = geo.getRowInfo(row).x;

    // check if calculated pad positions are equal to the real ones

    for (int pad = 0; pad < nPads; pad++) {
      const GlobalPadNumber p = mapper.globalPadNumber(PadPos(row, pad));
      const PadCentre& c = mapper.padCentre(p);
      double u = geo.convPadToU(row, pad);

      const double dx = x - c.X();
      const double dy = u - (-c.Y()); // diferent sign convention for Y coordinate in the map

      if (fabs(dx) >= 1.e-6 || fabs(dy) >= 1.e-5) {
        LOG(warning) << "wrong calculated pad position:"
                     << " row " << row << " pad " << pad << " x calc " << x << " x in map " << c.X() << " dx " << (x - c.X())
                     << " y calc " << u << " y in map " << -c.Y() << " dy " << dy << std::endl;
      }
      if (fabs(maxDx) < fabs(dx)) {
        maxDx = dx;
      }
      if (fabs(maxDy) < fabs(dy)) {
        maxDy = dy;
      }
    }
  }

  if (fabs(maxDx) >= 1.e-4 || fabs(maxDy) >= 1.e-4) {
    LOG(fatal) << "wrong calculated pad position:"
               << " max Dx " << maxDx << " max Dy " << maxDy << std::endl;
  }
}
} // namespace tpc
} // namespace o2
