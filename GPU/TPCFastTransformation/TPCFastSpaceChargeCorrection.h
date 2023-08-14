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

/// \file  TPCFastSpaceChargeCorrection.h
/// \brief Definition of TPCFastSpaceChargeCorrection class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTSPACECHARGECORRECTION_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTSPACECHARGECORRECTION_H

#include "Spline2D.h"
#include "TPCFastTransformGeo.h"
#include "FlatObject.h"
#include "GPUCommonDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

///
/// The TPCFastSpaceChargeCorrection class represents correction of nominal coordinates of TPC clusters
/// using best-fit splines
///
/// Row, U, V -> dX,dU,dV
///
/// The class is flat C structure. No virtual methods, no ROOT types are used.
///
class TPCFastSpaceChargeCorrection : public FlatObject
{
 public:
  ///
  /// \brief The struct contains necessary info for TPC padrow
  ///
  struct RowInfo {
    int splineScenarioID{0};      ///< scenario index (which of Spline2D splines to use)
    size_t dataOffsetBytes[3]{0}; ///< offset for the spline data withing a TPC slice
#ifndef GPUCA_ALIROOT_LIB
    ClassDefNV(RowInfo, 1);
#endif
  };

  struct RowActiveArea {
    float maxDriftLengthCheb[5]{0.};
    float vMax{0.};
    float cuMin{0.};
    float cuMax{0.};
    float cvMax{0.};
#ifndef GPUCA_ALIROOT_LIB
    ClassDefNV(RowActiveArea, 1);
#endif
  };

  struct SliceRowInfo {
    float gridV0{0.};           ///< V coordinate of the V-grid start
    float gridCorrU0{0.};       ///< U coordinate of the U-grid start for corrected U
    float gridCorrV0{0.};       ///< V coordinate of the V-grid start for corrected V
    float scaleCorrUtoGrid{0.}; ///< scale corrected U to U-grid coordinate
    float scaleCorrVtoGrid{0.}; ///< scale corrected V to V-grid coordinate
    RowActiveArea activeArea;
#ifndef GPUCA_ALIROOT_LIB
    ClassDefNV(SliceRowInfo, 1);
#endif
  };

  struct SliceInfo {
    float vMax{0.}; ///< Max value of V coordinate
#ifndef GPUCA_ALIROOT_LIB
    ClassDefNV(SliceInfo, 1);
#endif
  };

  typedef Spline2D<float, 3> SplineType;

  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCFastSpaceChargeCorrection();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneTo[In/Ex]ternalBuffer() instead
  TPCFastSpaceChargeCorrection(const TPCFastSpaceChargeCorrection&) CON_DELETE;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneTo[In/Ex]ternalBuffer() instead
  TPCFastSpaceChargeCorrection& operator=(const TPCFastSpaceChargeCorrection&) CON_DELETE;

  /// Destructor
  ~TPCFastSpaceChargeCorrection();

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  /// Memory alignment

  using FlatObject::getBufferAlignmentBytes;
  using FlatObject::getClassAlignmentBytes;

  /// Construction interface

  void cloneFromObject(const TPCFastSpaceChargeCorrection& obj, char* newFlatBufferPtr);
  void destroy();

  /// Making the data buffer external

  using FlatObject::releaseInternalBuffer;
  void moveBufferTo(char* newBufferPtr);

  /// Moving the class with its external buffer to another location

  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

  /// _______________  Construction interface  ________________________

  /// Starts the construction procedure, reserves temporary memory
  void startConstruction(const TPCFastTransformGeo& geo, int numberOfSplineScenarios);

  /// Initializes a TPC row
  void setRowScenarioID(int iRow, int iScenario);

  /// Sets approximation scenario
  void setSplineScenario(int scenarioIndex, const SplineType& spline);

  /// Finishes construction: puts everything to the flat buffer, releases temporary memory
  void finishConstruction();

  void constructWithNoCorrection(const TPCFastTransformGeo& geo);

  /// _______________  Initialization interface  ________________________

  /// Set no correction
  GPUd() void setNoCorrection();

  /// Sets the time stamp of the current calibaration
  GPUd() void setTimeStamp(long int v) { mTimeStamp = v; }

  /// Set safety marging for the interpolation around the TPC row.
  /// Outside of this area the interpolation returns the boundary values.
  GPUd() void setInterpolationSafetyMargin(float val) { fInterpolationSafetyMargin = val; }

  /// Gives const pointer to a spline
  GPUd() const SplineType& getSpline(int slice, int row) const;

  /// Gives pointer to a spline
  GPUd() SplineType& getSpline(int slice, int row);

  /// Gives pointer to spline data
  GPUd() float* getSplineData(int slice, int row, int iSpline = 0);

  /// Gives pointer to spline data
  GPUd() const float* getSplineData(int slice, int row, int iSpline = 0) const;

  /// _______________ The main method: cluster correction  _______________________
  ///
  GPUd() int getCorrection(int slice, int row, float u, float v, float& dx, float& du, float& dv) const;

  /// inverse correction: Corrected U and V -> coorrected X
  GPUd() void getCorrectionInvCorrectedX(int slice, int row, float corrU, float corrV, float& corrX) const;

  /// inverse correction: Corrected U and V -> uncorrected U and V
  GPUd() void getCorrectionInvUV(int slice, int row, float corrU, float corrV, float& nomU, float& nomV) const;

  /// maximal possible drift length of the active area
  GPUd() float getMaxDriftLength(int slice, int row, float pad) const;

  /// maximal possible drift length of the active area
  GPUd() float getMaxDriftLength(int slice, int row) const;

  /// maximal possible drift length of the active area
  GPUd() float getMaxDriftLength(int slice) const;

  /// _______________  Utilities  _______________________________________________

  /// shrink u,v coordinats to the TPC row area +/- fkInterpolationSafetyMargin
  GPUd() void schrinkUV(int slice, int row, float& u, float& v) const;

  /// shrink corrected u,v coordinats to the TPC row area +/- fkInterpolationSafetyMargin
  GPUd() void schrinkCorrectedUV(int slice, int row, float& corrU, float& corrV) const;

  /// convert u,v to internal grid coordinates
  GPUd() void convUVtoGrid(int slice, int row, float u, float v, float& gridU, float& gridV) const;

  /// convert u,v to internal grid coordinates
  GPUd() void convGridToUV(int slice, int row, float gridU, float gridV, float& u, float& v) const;

  /// convert corrected u,v to internal grid coordinates
  GPUd() void convCorrectedUVtoGrid(int slice, int row, float cu, float cv, float& gridU, float& gridV) const;

  /// TPC geometry information
  GPUd() const TPCFastTransformGeo& getGeometry() const
  {
    return mGeo;
  }

  /// Gives the time stamp of the current calibaration parameters
  long int getTimeStamp() const { return mTimeStamp; }

  /// Gives the interpolation safety marging  around the TPC row.
  GPUd() float getInterpolationSafetyMargin() const { return fInterpolationSafetyMargin; }

  /// Gives TPC row info
  GPUd() const RowInfo& getRowInfo(int row) const { return mRowInfoPtr[row]; }

  /// Gives TPC slice info
  GPUd() const SliceInfo& getSliceInfo(int slice) const
  {
    return mSliceInfo[slice];
  }

  /// Gives TPC slice info
  GPUd() SliceInfo& getSliceInfo(int slice)
  {
    return mSliceInfo[slice];
  }

  /// Gives TPC slice & row info
  GPUd() const SliceRowInfo& getSliceRowInfo(int slice, int row) const
  {
    return mSliceRowInfoPtr[mGeo.getNumberOfRows() * slice + row];
  }

  /// Gives TPC slice & row info
  GPUd() SliceRowInfo& getSliceRowInfo(int slice, int row)
  {
    return mSliceRowInfoPtr[mGeo.getNumberOfRows() * slice + row];
  }

#if !defined(GPUCA_GPUCODE)
  /// Print method
  void print() const;
  GPUh() double testInverse(bool prn = 0);
#endif

 private:
  /// relocate buffer pointers
  void relocateBufferPointers(const char* oldBuffer, char* newBuffer);
  /// release temporary memory used during construction
  void releaseConstructionMemory();

  /// temporary method with the an way of calculating 2D spline
  GPUd() int getCorrectionOld(int slice, int row, float u, float v, float& dx, float& du, float& dv) const;

  /// _______________  Data members  _______________________________________________

  /// _______________  Construction control  _______________________________________________

  RowInfo* mConstructionRowInfos = nullptr;     //! (transient!!) Temporary container of the row infos during construction
  SplineType* mConstructionScenarios = nullptr; //! (transient!!) Temporary container for spline scenarios

  /// _______________  Geometry  _______________________________________________

  TPCFastTransformGeo mGeo; ///< TPC geometry information

  int mNumberOfScenarios; ///< Number of approximation spline scenarios

  SliceInfo mSliceInfo[TPCFastTransformGeo::getNumberOfSlices()]; ///< SliceInfo array

  SplineType* mScenarioPtr;       //! (transient!!) pointer to spline scenarios
  RowInfo* mRowInfoPtr;           //! (transient!!) pointer to RowInfo array inside the mFlatBufferPtr buffer
  SliceRowInfo* mSliceRowInfoPtr; //! (transient!!) pointer to SliceRowInfo array inside the mFlatBufferPtr

  /// _______________  Calibration data  _______________________________________________

  long int mTimeStamp; ///< time stamp of the current calibration

  char* mSplineData[3]; //! (transient!!) pointer to the spline data in the flat buffer

  size_t mSliceDataSizeBytes[3]; ///< size of the data for one slice in the flat buffer

  float fInterpolationSafetyMargin{0.1}; // 10% area around the TPC row. Outside of this area the interpolation returns the boundary values.

#ifndef GPUCA_ALIROOT_LIB
  ClassDefNV(TPCFastSpaceChargeCorrection, 3);
#endif
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

GPUdi() const TPCFastSpaceChargeCorrection::SplineType& TPCFastSpaceChargeCorrection::getSpline(int slice, int row) const
{
  /// Gives const pointer to spline
  const RowInfo& rowInfo = mRowInfoPtr[row];
  return mScenarioPtr[rowInfo.splineScenarioID];
}

GPUdi() TPCFastSpaceChargeCorrection::SplineType& TPCFastSpaceChargeCorrection::getSpline(int slice, int row)
{
  /// Gives pointer to spline
  const RowInfo& rowInfo = mRowInfoPtr[row];
  return mScenarioPtr[rowInfo.splineScenarioID];
}

GPUdi() float* TPCFastSpaceChargeCorrection::getSplineData(int slice, int row, int iSpline)
{
  /// Gives pointer to spline data
  const RowInfo& rowInfo = mRowInfoPtr[row];
  return reinterpret_cast<float*>(mSplineData[iSpline] + mSliceDataSizeBytes[iSpline] * slice + rowInfo.dataOffsetBytes[iSpline]);
}

GPUdi() const float* TPCFastSpaceChargeCorrection::getSplineData(int slice, int row, int iSpline) const
{
  /// Gives pointer to spline data
  const RowInfo& rowInfo = mRowInfoPtr[row];
  return reinterpret_cast<float*>(mSplineData[iSpline] + mSliceDataSizeBytes[iSpline] * slice + rowInfo.dataOffsetBytes[iSpline]);
}

GPUdi() void TPCFastSpaceChargeCorrection::schrinkUV(int slice, int row, float& u, float& v) const
{
  /// shrink u,v coordinats to the TPC row area +/- fInterpolationSafetyMargin

  const TPCFastTransformGeo::RowInfo& rowInfo = mGeo.getRowInfo(row);

  float uWidth05 = rowInfo.getUwidth() * (0.5f + fInterpolationSafetyMargin);
  float vWidth = mGeo.getTPCzLength(slice);

  if (u < -uWidth05) {
    u = -uWidth05;
  }
  if (u > uWidth05) {
    u = uWidth05;
  }
  if (v < -0.1f * vWidth) {
    v = -0.1f * vWidth;
  }
  if (v > 1.1f * vWidth) {
    v = 1.1f * vWidth;
  }
}

GPUdi() void TPCFastSpaceChargeCorrection::schrinkCorrectedUV(int slice, int row, float& corrU, float& corrV) const
{
  /// shrink corrected u,v coordinats to the TPC row area +/- fInterpolationSafetyMargin

  const TPCFastTransformGeo::RowInfo& rowInfo = mGeo.getRowInfo(row);
  const SliceRowInfo& sliceRowInfo = getSliceRowInfo(slice, row);

  float uMargin = fInterpolationSafetyMargin * rowInfo.getUwidth();
  float vMargin = fInterpolationSafetyMargin * mGeo.getTPCzLength(slice);

  if (corrU < sliceRowInfo.activeArea.cuMin - uMargin) {
    corrU = sliceRowInfo.activeArea.cuMin - uMargin;
  }

  if (corrU > sliceRowInfo.activeArea.cuMax + uMargin) {
    corrU = sliceRowInfo.activeArea.cuMax + uMargin;
  }

  if (corrV < 0.f - vMargin) {
    corrV = 0.f - vMargin;
  }

  if (corrV > sliceRowInfo.activeArea.cvMax + vMargin) {
    corrV = sliceRowInfo.activeArea.cvMax + vMargin;
  }
}

GPUdi() void TPCFastSpaceChargeCorrection::convUVtoGrid(int slice, int row, float u, float v, float& gu, float& gv) const
{
  // TODO optimise !!!
  gu = 0.;
  gv = 0.;

  schrinkUV(slice, row, u, v);

  const SliceRowInfo& info = getSliceRowInfo(slice, row);
  const SplineType& spline = getSpline(slice, row);

  float su0 = 0., sv0 = 0.;
  mGeo.convUVtoScaledUV(slice, row, u, info.gridV0, su0, sv0);
  mGeo.convUVtoScaledUV(slice, row, u, v, gu, gv);

  gv = (gv - sv0) / (1. - sv0);
  gu *= spline.getGridX1().getUmax();
  gv *= spline.getGridX2().getUmax();
}

GPUdi() void TPCFastSpaceChargeCorrection::convGridToUV(int slice, int row, float gridU, float gridV, float& u, float& v) const
{
  // TODO optimise
  /// convert u,v to internal grid coordinates
  float su0 = 0., sv0 = 0.;
  const SliceRowInfo& info = getSliceRowInfo(slice, row);
  const SplineType& spline = getSpline(slice, row);
  mGeo.convUVtoScaledUV(slice, row, 0., info.gridV0, su0, sv0);
  float su = gridU / spline.getGridX1().getUmax();
  float sv = sv0 + gridV / spline.getGridX2().getUmax() * (1. - sv0);
  mGeo.convScaledUVtoUV(slice, row, su, sv, u, v);
}

GPUdi() void TPCFastSpaceChargeCorrection::convCorrectedUVtoGrid(int slice, int row, float corrU, float corrV, float& gridU, float& gridV) const
{
  schrinkCorrectedUV(slice, row, corrU, corrV);

  const SliceRowInfo& sliceRowInfo = getSliceRowInfo(slice, row);

  gridU = (corrU - sliceRowInfo.gridCorrU0) * sliceRowInfo.scaleCorrUtoGrid;
  gridV = (corrV - sliceRowInfo.gridCorrV0) * sliceRowInfo.scaleCorrVtoGrid;
}

GPUdi() int TPCFastSpaceChargeCorrection::getCorrection(int slice, int row, float u, float v, float& dx, float& du, float& dv) const
{
  const SplineType& spline = getSpline(slice, row);
  const float* splineData = getSplineData(slice, row);
  float gridU = 0, gridV = 0;
  convUVtoGrid(slice, row, u, v, gridU, gridV);
  float dxuv[3];
  spline.interpolateU(splineData, gridU, gridV, dxuv);
  dx = dxuv[0];
  du = dxuv[1];
  dv = dxuv[2];
  return 0;
}

GPUdi() int TPCFastSpaceChargeCorrection::getCorrectionOld(int slice, int row, float u, float v, float& dx, float& du, float& dv) const
{
  const SplineType& spline = getSpline(slice, row);
  const float* splineData = getSplineData(slice, row);
  float gridU = 0, gridV = 0;
  convUVtoGrid(slice, row, u, v, gridU, gridV);
  float dxuv[3];
  spline.interpolateUold(splineData, gridU, gridV, dxuv);
  dx = dxuv[0];
  du = dxuv[1];
  dv = dxuv[2];
  return 0;
}

GPUdi() void TPCFastSpaceChargeCorrection::getCorrectionInvCorrectedX(
  int slice, int row, float corrU, float corrV, float& x) const
{
  float gridU, gridV;
  convCorrectedUVtoGrid(slice, row, corrU, corrV, gridU, gridV);

  const Spline2D<float, 1>& spline = reinterpret_cast<const Spline2D<float, 1>&>(getSpline(slice, row));
  const float* splineData = getSplineData(slice, row, 1);
  float dx = 0;
  spline.interpolateU(splineData, gridU, gridV, &dx);
  x = mGeo.getRowInfo(row).x + dx;
}

GPUdi() void TPCFastSpaceChargeCorrection::getCorrectionInvUV(
  int slice, int row, float corrU, float corrV, float& nomU, float& nomV) const
{
  float gridU, gridV;
  convCorrectedUVtoGrid(slice, row, corrU, corrV, gridU, gridV);

  const Spline2D<float, 2>& spline = reinterpret_cast<const Spline2D<float, 2>&>(getSpline(slice, row));
  const float* splineData = getSplineData(slice, row, 2);

  float duv[2];
  spline.interpolateU(splineData, gridU, gridV, duv);
  nomU = corrU - duv[0];
  nomV = corrV - duv[1];
}

GPUdi() float TPCFastSpaceChargeCorrection::getMaxDriftLength(int slice, int row, float pad) const
{
  const RowActiveArea& area = getSliceRowInfo(slice, row).activeArea;
  const float* c = area.maxDriftLengthCheb;
  float x = -1.f + 2.f * pad / mGeo.getRowInfo(row).maxPad;
  float y = c[0] + c[1] * x;
  float f0 = 1.;
  float f1 = x;
  x *= 2.f;
  for (int i = 2; i < 5; i++) {
    double f = x * f1 - f0;
    y += c[i] * f;
    f0 = f1;
    f1 = f;
  }
  return y;
}

GPUdi() float TPCFastSpaceChargeCorrection::getMaxDriftLength(int slice, int row) const
{
  return getSliceRowInfo(slice, row).activeArea.vMax;
}

GPUdi() float TPCFastSpaceChargeCorrection::getMaxDriftLength(int slice) const
{
  return getSliceInfo(slice).vMax;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
