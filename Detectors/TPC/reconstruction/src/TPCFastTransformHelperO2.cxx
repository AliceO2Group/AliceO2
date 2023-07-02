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

std::unique_ptr<TPCFastTransform> TPCFastTransformHelperO2::create(Long_t TimeStamp, const TPCFastSpaceChargeCorrection& correction)
{
  /// initializes TPCFastTransform object

  // init geometry

  if (!mIsInitialized) {
    init();
  }

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

  updateCalibration(fastTransform, TimeStamp);

  return std::move(fastTransformPtr);
}

std::unique_ptr<TPCFastTransform> TPCFastTransformHelperO2::create(Long_t TimeStamp)
{
  /// initializes TPCFastTransform object

  // init geometry

  if (!mIsInitialized) {
    init();
  }

  TPCFastSpaceChargeCorrection correction;
  correction.constructWithNoCorrection(mGeo);

  return create(TimeStamp, correction);
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
