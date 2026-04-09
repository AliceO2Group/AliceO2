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

#ifndef GPUCA_STANDALONE
#include "TPCBase/Mapper.h"
#include "TPCBase/PadRegionInfo.h"
#endif
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCBase/Sector.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCFastTransform.h"
#include "GPUTPCGeometry.h"
#include <GPUCommonLogger.h>

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

  const GPUTPCGeometry geo;

  const int nRows = geo.NROWS;

  mGeo.startConstruction(nRows);
  mGeo.setTPCzLength(geo.TPCLength());

  for (int iRow = 0; iRow < nRows; iRow++) {
    mGeo.setTPCrow(iRow, geo.Row2X(iRow), geo.NPads(iRow), geo.PadWidth(iRow));
  }

  mGeo.finishConstruction();

#ifndef GPUCA_STANDALONE
  // check if calculated pad geometry is consistent with the map
  testGeometry(mGeo);
#endif

  mIsInitialized = 1;
}

std::unique_ptr<TPCFastTransform> TPCFastTransformHelperO2::create(int64_t TimeStamp, const TPCFastSpaceChargeCorrection& correction)
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
    const long int initTimeStamp = -1;
    fastTransform.setCalibration1(initTimeStamp, t0, vDrift);

    fastTransform.finishConstruction();
  }

  updateCalibration(fastTransform, TimeStamp);

  return fastTransformPtr;
}

std::unique_ptr<TPCFastTransform> TPCFastTransformHelperO2::create(int64_t TimeStamp)
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

template <typename T>
int TPCFastTransformHelperO2::updateCalibrationImpl(T& fastTransform, int64_t TimeStamp, float vDriftFactor, float vDriftRef, float driftTimeOffset)
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

  auto& elParam = ParameterElectronics::Instance();
  // start the initialization

  fastTransform.setTimeStamp(TimeStamp);
  if (vDriftRef == 0) {
    vDriftRef = ParameterGas::Instance().DriftV;
  }
  const double vDrift = elParam.ZbinWidth * vDriftRef * vDriftFactor; // cm/timebin

  // fast transform formula:
  // L = (t-t0)*mVdrift
  // Z = Z(L)
  // spline corrections for xyz

  const double t0 = (driftTimeOffset + elParam.getAverageShapingTime()) / elParam.ZbinWidth;

  fastTransform.setCalibration1(TimeStamp, t0, vDrift);

  return 0;
}

#ifndef GPUCA_STANDALONE
void TPCFastTransformHelperO2::testGeometry(const TPCFastTransformGeo& geo) const
{
  const Mapper& mapper = Mapper::instance();

  if (geo.getNumberOfSectors() != Sector::MAXSECTOR) {
    LOG(fatal) << "Wrong number of sectors :" << geo.getNumberOfSectors() << " instead of " << Sector::MAXSECTOR << std::endl;
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

      float y, z;
      geo.convPadDriftLengthToLocal(0, row, pad, 0., y, z);

      const double dx = x - c.X();
      const double dy = y - (-c.Y()); // diferent sign convention for Y coordinate in the map

      if (fabs(dx) >= 1.e-6 || fabs(dy) >= 1.e-5) {
        LOG(warning) << "wrong calculated pad position:"
                     << " row " << row << " pad " << pad << " x calc " << x << " x in map " << c.X() << " dx " << (x - c.X())
                     << " y calc " << y << " y in map " << -c.Y() << " dy " << dy << std::endl;
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
#endif

template int TPCFastTransformHelperO2::updateCalibrationImpl(TPCFastTransform&, int64_t, float, float, float);
template int TPCFastTransformHelperO2::updateCalibrationImpl(TPCFastTransformPOD&, int64_t, float, float, float);

} // namespace tpc
} // namespace o2
