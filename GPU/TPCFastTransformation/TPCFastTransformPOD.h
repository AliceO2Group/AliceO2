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

/// \file  TPCFastTransformPOD.h
/// \brief POD correction map
///
/// \author  ruben.shahoayn@cern.ch

#ifndef ALICEO2_GPU_TPCFastTransformPOD_H
#define ALICEO2_GPU_TPCFastTransformPOD_H

#include "TPCFastTransform.h"

/*
Binary buffer should be cast to TPCFastTransformPOD class using static TPCFastTransformPOD& t = get(buffer); method,
so that the its head becomes `this` pointer of the object.

First we have all the fixed size data members mentioned explicitly. Part of them is duplicating fixed size
data members of TPCFastSpaceChargeCorrection but those starting with mOffs... provide the offset in bytes
(wrt this) for dynamic data which cannot be declared as data member explicitly (since we cannot have any
pointer except `this`) but obtained via getters using stored offsets wrt `this`.
This is followed dynamic part itself.

dynamic part layout:
1) size_t[ mNumberOfScenarios ] array starting at offset mOffsScenariosOffsets, each element is the offset
of distict spline object (scenario in TPCFastSpaceChargeCorrection)
2) size_t[ mNSplineIDs ] array starting at offset mOffsSplineDataOffsets, each element is the offset of the
beginning of splines data for give splineID

*/

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class TPCFastTransformPOD
{
 public:
  using SplineType = TPCFastSpaceChargeCorrection::SplineType;
  using RowInfo = TPCFastSpaceChargeCorrection::RowInfo;
  using RowActiveArea = TPCFastSpaceChargeCorrection::RowActiveArea;
  using SliceRowInfo = TPCFastSpaceChargeCorrection::SliceRowInfo;
  using SliceInfo = TPCFastSpaceChargeCorrection::SliceInfo;

  /// convert prefilled buffer to TPCFastTransformPOD
  GPUd() static const TPCFastTransformPOD& get(const char* head) { return *reinterpret_cast<const TPCFastTransformPOD*>(head); }

  GPUd() int getApplyCorrections() const { return mApplyCorrections; }
  GPUd() void setApplyCorrections(bool v = true) { mApplyCorrections = v; }

  /// _______________ high level methods a la TPCFastTransform  _______________________
  ///
  GPUd() void Transform(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime = 0) const;
  GPUd() void TransformXYZ(int slice, int row, float& x, float& y, float& z) const;

  /// Transformation in the time frame
  GPUd() void TransformInTimeFrame(int slice, int row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const;
  GPUd() void TransformInTimeFrame(int slice, float time, float& z, float maxTimeBin) const;

  /// Inverse transformation
  GPUd() void InverseTransformInTimeFrame(int slice, int row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const;

  /// Inverse transformation: Transformed Y and Z -> transformed X
  GPUd() void InverseTransformYZtoX(int slice, int row, float y, float z, float& x) const;

  /// Inverse transformation: Transformed Y and Z -> Y and Z, transformed w/o space charge correction
  GPUd() void InverseTransformYZtoNominalYZ(int slice, int row, float y, float z, float& ny, float& nz) const;

  /// Inverse transformation: Transformed X, Y and Z -> X, Y and Z, transformed w/o space charge correction
  GPUd() void InverseTransformXYZtoNominalXYZ(int slice, int row, float x, float y, float z, float& nx, float& ny, float& nz) const;

  /// Ideal transformation with Vdrift only - without calibration
  GPUd() void TransformIdeal(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime) const;
  GPUd() void TransformIdealZ(int slice, float time, float& z, float vertexTime) const;

  GPUd() void convPadTimeToUV(int slice, int row, float pad, float time, float& u, float& v, float vertexTime) const;
  GPUd() void convPadTimeToUVinTimeFrame(int slice, int row, float pad, float time, float& u, float& v, float maxTimeBin) const;
  GPUd() void convTimeToVinTimeFrame(int slice, float time, float& v, float maxTimeBin) const;

  GPUd() void convUVtoPadTime(int slice, int row, float u, float v, float& pad, float& time, float vertexTime) const;
  GPUd() void convUVtoPadTimeInTimeFrame(int slice, int row, float u, float v, float& pad, float& time, float maxTimeBin) const;
  GPUd() void convVtoTime(float v, float& time, float vertexTime) const;

  GPUd() float convTimeToZinTimeFrame(int slice, float time, float maxTimeBin) const;
  GPUd() float convZtoTimeInTimeFrame(int slice, float z, float maxTimeBin) const;
  GPUd() float convDeltaTimeToDeltaZinTimeFrame(int slice, float deltaTime) const;
  GPUd() float convDeltaZtoDeltaTimeInTimeFrame(int slice, float deltaZ) const;
  GPUd() float convDeltaZtoDeltaTimeInTimeFrameAbs(float deltaZ) const;
  GPUd() float convZOffsetToVertexTime(int slice, float zOffset, float maxTimeBin) const;
  GPUd() float convVertexTimeToZOffset(int slice, float vertexTime, float maxTimeBin) const;

  GPUd() void getTOFcorrection(int slice, int row, float x, float y, float z, float& dz) const;

  /// _______________ methods a la TPCFastSpaceChargeCorrection: cluster correction  _______________________
  ///
  GPUd() int getCorrection(int slice, int row, float u, float v, float& dx, float& du, float& dv) const;

  /// temporary method with the an way of calculating 2D spline
  GPUd() int getCorrectionOld(int slice, int row, float u, float v, float& dx, float& du, float& dv) const;

  /// inverse correction: Corrected U and V -> coorrected X
  GPUd() void getCorrectionInvCorrectedX(int slice, int row, float corrU, float corrV, float& corrX) const;

  /// inverse correction: Corrected U and V -> uncorrected U and V
  GPUd() void getCorrectionInvUV(int slice, int row, float corrU, float corrV, float& nomU, float& nomV) const;

  /// maximal possible drift length of the active area
  GPUd() float getMaxDriftLength(int slice, int row, float pad) const;

  /// maximal possible drift length of the active area
  GPUd() float getMaxDriftLength(int slice, int row) const { return getSliceRowInfo(slice, row).activeArea.vMax; }

  /// maximal possible drift length of the active area
  GPUd() float getMaxDriftLength(int slice) const { return getSliceInfo(slice).vMax; }

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int slice, int row, float pad) const;

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int slice, int row) const;

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int slice) const;

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
  GPUd() const TPCFastTransformGeo& getGeometry() const { return mGeo; }

  /// Gives its own size including dynamic part
  GPUd() size_t size() const { return mTotalSize; }

  /// Gives the time stamp of the current calibaration parameters
  GPUd() long int getTimeStamp() const { return mTimeStamp; }

  /// Gives the interpolation safety marging  around the TPC row.
  GPUd() float getInterpolationSafetyMargin() const { return mInterpolationSafetyMargin; }

  /// Return mVDrift in cm / time bin
  GPUd() float getVDrift() const { return mVdrift; }

  /// Return T0 in time bin units
  GPUd() float getT0() const { return mT0; }

  /// Return VdriftCorrY in time_bin / cn
  GPUd() float getVdriftCorrY() const { return mVdriftCorrY; }

  /// Return LdriftCorr offset in cm
  GPUd() float getLdriftCorr() const { return mLdriftCorr; }

  /// Return TOF correction
  GPUd() float getTOFCorr() const { return mTOFcorr; }

  /// Return nominal PV Z position
  GPUd() float getPrimVtxZ() const { return mPrimVtxZ; }

  /// Sets the time stamp of the current calibaration
  GPUd() void setTimeStamp(long int v) { mTimeStamp = v; }

  /// Set safety marging for the interpolation around the TPC row.
  /// Outside of this area the interpolation returns the boundary values.
  GPUd() void setInterpolationSafetyMargin(float val) { mInterpolationSafetyMargin = val; }

  /// Gives TPC row info
  GPUd() const RowInfo& getRowInfo(int row) const { return mRowInfo[row]; }

  /// Gives TPC slice info
  GPUd() const SliceInfo& getSliceInfo(int slice) const { return mSliceInfo[slice]; }

  /// Gives a reference to a spline
  GPUd() const SplineType& getSpline(int slice, int row) const { return *reinterpret_cast<const SplineType*>(getThis() + getScenarioOffset(getRowInfo(row).splineScenarioID)); }

  /// Gives pointer to spline data
  GPUd() const float* getSplineData(int slice, int row, int iSpline = 0) const { return reinterpret_cast<const float*>(getThis() + mSplineDataOffsets[slice][iSpline] + getRowInfo(row).dataOffsetBytes[iSpline]); }

  /// Gives TPC slice & row info
  GPUd() const SliceRowInfo& getSliceRowInfo(int slice, int row) const { return mSliceRowInfo[NROWS * slice + row]; }

#if !defined(GPUCA_GPUCODE)
  /// Create POD transform from old flat-buffer one. Provided vector will serve as a buffer
  template <typename V>
  static TPCFastTransformPOD& create(V& destVector, const TPCFastTransform& src);

  /// create filling only part corresponding to TPCFastSpaceChargeCorrection. Data members coming from TPCFastTransform (e.g. VDrift, T0..) are not set
  template <typename V>
  static TPCFastTransformPOD& create(V& destVector, const TPCFastSpaceChargeCorrection& src);

  bool test(const TPCFastTransform& src, int npoints = 100000) const { return test(src.getCorrection(), npoints); }
  bool test(const TPCFastSpaceChargeCorrection& origCorr, int npoints = 100000) const;
#endif

  static constexpr int NROWS = 152;
  static constexpr int NSLICES = TPCFastTransformGeo::getNumberOfSlices();
  static constexpr int NSplineIDs = 3; ///< number of spline data sets for each slice/row

 private:
#if !defined(GPUCA_GPUCODE)
  static constexpr size_t AlignmentBytes = 8;
  static size_t alignOffset(size_t offs)
  {
    auto res = offs % AlignmentBytes;
    return res ? offs + (AlignmentBytes - res) : offs;
  }
  static size_t estimateSize(const TPCFastTransform& src) { return estimateSize(src.getCorrection()); }
  static size_t estimateSize(const TPCFastSpaceChargeCorrection& origCorr);
  static TPCFastTransformPOD& create(char* buff, size_t buffSize, const TPCFastTransform& src);
  static TPCFastTransformPOD& create(char* buff, size_t buffSize, const TPCFastSpaceChargeCorrection& src);

  ///< get address to which the offset in bytes must be added to arrive to particular dynamic part
  GPUd() const char* getThis() const { return reinterpret_cast<const char*>(this); }
  GPUd() static TPCFastTransformPOD& getNonConst(char* head) { return *reinterpret_cast<TPCFastTransformPOD*>(head); }

#endif

  /// Gives non-const TPC row info
  GPUd() RowInfo& getRowInfo(int row) { return mRowInfo[row]; }

  ///< return offset of the spline object start (equivalent of mScenarioPtr in the TPCFastSpaceChargeCorrection)
  GPUd() const size_t getScenarioOffset(int s) const { return (reinterpret_cast<const size_t*>(getThis() + mOffsScenariosOffsets))[s]; }

  GPUd() void TransformInternal(int slice, int row, float& u, float& v, float& x) const;

  bool mApplyCorrections{};                                                        ///< flag to apply corrections
  int mNumberOfScenarios{};                                                        ///< Number of approximation spline scenarios
  size_t mTotalSize{};                                                             ///< total size of the buffer
  size_t mOffsScenariosOffsets{};                                                  ///< start of the array of mNumberOfScenarios offsets for each type of spline
  size_t mSplineDataOffsets[TPCFastTransformGeo::getNumberOfSlices()][NSplineIDs]; ///< start of data for each slice and iSpline data
  long int mTimeStamp{};                                                           ///< time stamp of the current calibration
  float mInterpolationSafetyMargin{0.1f};                                          // 10% area around the TPC row. Outside of this area the interpolation returns the boundary values.

  /// _____ Parameters for drift length calculation ____
  ///
  /// t = (float) time bin, y = global y
  ///
  /// L(t,y) = (t-mT0)*(mVdrift + mVdriftCorrY*y ) + mLdriftCorr  ____
  ///
  float mT0;          ///< T0 in [time bin]
  float mVdrift;      ///< VDrift in  [cm/time bin]
  float mVdriftCorrY; ///< VDrift correction for global Y[cm] in [1/time bin]
  float mLdriftCorr;  ///< drift length correction in [cm]

  /// A coefficient for Time-Of-Flight correction: drift length -= EstimatedDistanceToVtx[cm]*mTOFcorr
  ///
  /// Since this correction requires a knowledge of the spatial position, it is appied after mCorrection,
  /// not on the drift length but directly on V coordinate.
  ///
  /// mTOFcorr == mVdrift/(speed of light)
  ///
  float mTOFcorr;

  float mPrimVtxZ; ///< Z of the primary vertex, needed for the Time-Of-Flight correction

  TPCFastTransformGeo mGeo;                                                                     ///< TPC geometry information
  TPCFastSpaceChargeCorrection::SliceInfo mSliceInfo[TPCFastTransformGeo::getNumberOfSlices()]; ///< SliceInfo array
  RowInfo mRowInfo[NROWS];
  SliceRowInfo mSliceRowInfo[NROWS * TPCFastTransformGeo::getNumberOfSlices()];

  ClassDefNV(TPCFastTransformPOD, 0);
};

GPUdi() int TPCFastTransformPOD::getCorrection(int slice, int row, float u, float v, float& dx, float& du, float& dv) const
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

GPUdi() int TPCFastTransformPOD::getCorrectionOld(int slice, int row, float u, float v, float& dx, float& du, float& dv) const
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

GPUdi() void TPCFastTransformPOD::getCorrectionInvCorrectedX(int slice, int row, float corrU, float corrV, float& x) const
{
  float gridU, gridV;
  convCorrectedUVtoGrid(slice, row, corrU, corrV, gridU, gridV);

  const Spline2D<float, 1>& spline = reinterpret_cast<const Spline2D<float, 1>&>(getSpline(slice, row));
  const float* splineData = getSplineData(slice, row, 1);
  float dx = 0;
  spline.interpolateU(splineData, gridU, gridV, &dx);
  x = mGeo.getRowInfo(row).x + dx;
}

GPUdi() void TPCFastTransformPOD::getCorrectionInvUV(int slice, int row, float corrU, float corrV, float& nomU, float& nomV) const
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

GPUdi() float TPCFastTransformPOD::getMaxDriftLength(int slice, int row, float pad) const
{
  const RowActiveArea& area = getSliceRowInfo(slice, row).activeArea;
  const float* c = area.maxDriftLengthCheb;
  float x = -1.f + 2.f * pad / mGeo.getRowInfo(row).maxPad;
  float y = c[0] + c[1] * x;
  float f0 = 1.f;
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

GPUdi() void TPCFastTransformPOD::schrinkUV(int slice, int row, float& u, float& v) const
{
  /// shrink u,v coordinats to the TPC row area +/- mInterpolationSafetyMargin

  float uWidth05 = mGeo.getRowInfo(row).getUwidth() * (0.5f + mInterpolationSafetyMargin);
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

GPUdi() void TPCFastTransformPOD::schrinkCorrectedUV(int slice, int row, float& corrU, float& corrV) const
{
  /// shrink corrected u,v coordinats to the TPC row area +/- mInterpolationSafetyMargin

  const SliceRowInfo& sliceRowInfo = getSliceRowInfo(slice, row);

  float uMargin = mInterpolationSafetyMargin * mGeo.getRowInfo(row).getUwidth();
  float vMargin = mInterpolationSafetyMargin * mGeo.getTPCzLength(slice);

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

GPUdi() void TPCFastTransformPOD::convUVtoGrid(int slice, int row, float u, float v, float& gu, float& gv) const
{
  // TODO optimise !!!
  gu = 0.f;
  gv = 0.f;

  schrinkUV(slice, row, u, v);

  const SliceRowInfo& info = getSliceRowInfo(slice, row);
  const SplineType& spline = getSpline(slice, row);

  float su0 = 0.f, sv0 = 0.f;
  mGeo.convUVtoScaledUV(slice, row, u, info.gridV0, su0, sv0);
  mGeo.convUVtoScaledUV(slice, row, u, v, gu, gv);

  gv = (gv - sv0) / (1.f - sv0);
  gu *= spline.getGridX1().getUmax();
  gv *= spline.getGridX2().getUmax();
}

GPUdi() void TPCFastTransformPOD::convGridToUV(int slice, int row, float gridU, float gridV, float& u, float& v) const
{
  // TODO optimise
  /// convert u,v to internal grid coordinates
  float su0 = 0.f, sv0 = 0.f;
  const SliceRowInfo& info = getSliceRowInfo(slice, row);
  const SplineType& spline = getSpline(slice, row);
  mGeo.convUVtoScaledUV(slice, row, 0.f, info.gridV0, su0, sv0);
  float su = gridU / spline.getGridX1().getUmax();
  float sv = sv0 + gridV / spline.getGridX2().getUmax() * (1.f - sv0);
  mGeo.convScaledUVtoUV(slice, row, su, sv, u, v);
}

GPUdi() void TPCFastTransformPOD::convCorrectedUVtoGrid(int slice, int row, float corrU, float corrV, float& gridU, float& gridV) const
{
  schrinkCorrectedUV(slice, row, corrU, corrV);

  const SliceRowInfo& sliceRowInfo = getSliceRowInfo(slice, row);

  gridU = (corrU - sliceRowInfo.gridCorrU0) * sliceRowInfo.scaleCorrUtoGrid;
  gridV = (corrV - sliceRowInfo.gridCorrV0) * sliceRowInfo.scaleCorrVtoGrid;
}

#if !defined(GPUCA_GPUCODE)
/// Create POD transform from old flat-buffer one. Provided vector will serve as a buffer
template <typename V>
TPCFastTransformPOD& TPCFastTransformPOD::create(V& destVector, const TPCFastTransform& src)
{
  const auto& origCorr = src.getCorrection();
  size_t estSize = estimateSize(src);
  destVector.resize(estSize); // allocate exact size
  LOGP(debug, "OrigCorrSize:{} SelfSize: {} Estimated POS size: {}", src.getCorrection().getFlatBufferSize(), sizeof(TPCFastTransformPOD), estSize);
  char* base = destVector.data();
  return create(destVector.data(), destVector.size(), src);
}

template <typename V>
TPCFastTransformPOD& TPCFastTransformPOD::create(V& destVector, const TPCFastSpaceChargeCorrection& origCorr)
{
  // create filling only part corresponding to TPCFastSpaceChargeCorrection. Data members coming from TPCFastTransform (e.g. VDrift, T0..) are not set
  size_t estSize = estimateSize(origCorr);
  destVector.resize(estSize); // allocate exact size
  LOGP(debug, "OrigCorrSize:{} SelfSize: {} Estimated POS size: {}", origCorr.getFlatBufferSize(), sizeof(TPCFastTransformPOD), estSize);
  char* base = destVector.data();
  return create(destVector.data(), destVector.size(), origCorr);
}
#endif

GPUdi() void TPCFastTransformPOD::Transform(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a slice
  /// taking calibration + alignment into account.
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  x = rowInfo.x;
  float u = 0, v = 0;
  convPadTimeToUV(slice, row, pad, time, u, v, vertexTime);
  TransformInternal(slice, row, u, v, x);
  getGeometry().convUVtoLocal(slice, u, v, y, z);
  float dzTOF = 0;
  getTOFcorrection(slice, row, x, y, z, dzTOF);
  z += dzTOF;
}

GPUdi() void TPCFastTransformPOD::TransformXYZ(int slice, int row, float& x, float& y, float& z) const
{
  float u, v;
  getGeometry().convLocalToUV(slice, y, z, u, v);
  TransformInternal(slice, row, u, v, x);
  getGeometry().convUVtoLocal(slice, u, v, y, z);
  float dzTOF = 0;
  getTOFcorrection(slice, row, x, y, z, dzTOF);
  z += dzTOF;
}

GPUdi() void TPCFastTransformPOD::TransformInTimeFrame(int slice, float time, float& z, float maxTimeBin) const
{
  float v = 0;
  convTimeToVinTimeFrame(slice, time, v, maxTimeBin);
  getGeometry().convVtoLocal(slice, v, z);
}

GPUdi() void TPCFastTransformPOD::TransformInTimeFrame(int slice, int row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const
{
  /// _______________ Special cluster transformation for a time frame _______________________
  ///
  /// Same as Transform(), but clusters are shifted in z such, that Z(maxTimeBin)==0
  /// Corrections and Time-Of-Flight correction are not alpplied.
  ///
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  x = rowInfo.x;
  float u = 0, v = 0;
  convPadTimeToUVinTimeFrame(slice, row, pad, time, u, v, maxTimeBin);
  getGeometry().convUVtoLocal(slice, u, v, y, z);
}

GPUdi() void TPCFastTransformPOD::InverseTransformInTimeFrame(int slice, int row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const
{
  /// Inverse transformation to TransformInTimeFrame
  float u = 0, v = 0;
  getGeometry().convLocalToUV(slice, y, z, u, v);
  convUVtoPadTimeInTimeFrame(slice, row, u, v, pad, time, maxTimeBin);
}

GPUdi() void TPCFastTransformPOD::InverseTransformYZtoX(int slice, int row, float y, float z, float& x) const
{
  /// Transformation y,z -> x
  float u = 0, v = 0;
  getGeometry().convLocalToUV(slice, y, z, u, v);
  if (mApplyCorrections) {
    getCorrectionInvCorrectedX(slice, row, u, v, x);
  } else {
    x = getGeometry().getRowInfo(row).x; // corrections are disabled
  }
  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_InverseTransformYZtoX").data()
                                                                                       << "slice=" << slice
                                                                                       << "row=" << row
                                                                                       << "y=" << y
                                                                                       << "z=" << z
                                                                                       << "x=" << x
                                                                                       << "v=" << v
                                                                                       << "u=" << u
                                                                                       << "\n";
  })
}

GPUdi() void TPCFastTransformPOD::InverseTransformYZtoNominalYZ(int slice, int row, float y, float z, float& ny, float& nz) const
{
  /// Transformation y,z -> x
  float u = 0, v = 0, un = 0, vn = 0;
  getGeometry().convLocalToUV(slice, y, z, u, v);
  if (mApplyCorrections) {
    getCorrectionInvUV(slice, row, u, v, un, vn);
  } else {
    un = u;
    vn = v;
  }
  getGeometry().convUVtoLocal(slice, un, vn, ny, nz);
  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_InverseTransformYZtoNominalYZ").data()
                                                                                       << "slice=" << slice
                                                                                       << "row=" << row
                                                                                       << "y=" << y
                                                                                       << "z=" << z
                                                                                       << "ny=" << ny
                                                                                       << "nz=" << nz
                                                                                       << "u=" << u
                                                                                       << "v=" << v
                                                                                       << "un=" << un
                                                                                       << "vn=" << vn
                                                                                       << "\n";
  })
}

GPUdi() void TPCFastTransformPOD::InverseTransformXYZtoNominalXYZ(int slice, int row, float x, float y, float z, float& nx, float& ny, float& nz) const
{
  /// Inverse transformation: Transformed X, Y and Z -> X, Y and Z, transformed w/o space charge correction
  int row2 = row + 1;
  if (row2 >= getGeometry().getNumberOfRows()) {
    row2 = row - 1;
  }
  float nx1, ny1, nz1; // nominal coordinates for row
  float nx2, ny2, nz2; // nominal coordinates for row2
  nx1 = getGeometry().getRowInfo(row).x;
  nx2 = getGeometry().getRowInfo(row2).x;
  InverseTransformYZtoNominalYZ(slice, row, y, z, ny1, nz1);
  InverseTransformYZtoNominalYZ(slice, row2, y, z, ny2, nz2);
  float c1 = (nx2 - nx) / (nx2 - nx1);
  float c2 = (nx - nx1) / (nx2 - nx1);
  nx = x;
  ny = (ny1 * c1 + ny2 * c2);
  nz = (nz1 * c1 + nz2 * c2);
}

GPUdi() void TPCFastTransformPOD::TransformInternal(int slice, int row, float& u, float& v, float& x) const
{
  if (mApplyCorrections) {
    float dx = 0.f, du = 0.f, dv = 0.f;
    getCorrection(slice, row, u, v, dx, du, dv);
    GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
      float ly, lz;
      getGeometry().convUVtoLocal(slice, u, v, ly, lz);
      float gx, gy, gz;
      getGeometry().convLocalToGlobal(slice, x, ly, lz, gx, gy, gz);
      float lyT, lzT;
      float uCorr = u + du;
      float vCorr = v + dv;
      float lxT = x + dx;
      getGeometry().convUVtoLocal(slice, uCorr, vCorr, lyT, lzT);
      float invYZtoX;
      InverseTransformYZtoX(slice, row, lyT, lzT, invYZtoX);
      float YZtoNominalY;
      float YZtoNominalZ;
      InverseTransformYZtoNominalYZ(slice, row, lyT, lzT, YZtoNominalY, YZtoNominalZ);
      o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_Transform").data()
                                                                                         // corrections in x, u, v
                                                                                         << "dx=" << dx
                                                                                         << "du=" << du
                                                                                         << "dv=" << dv
                                                                                         << "v=" << v
                                                                                         << "u=" << u
                                                                                         << "row=" << row
                                                                                         << "slice=" << slice
                                                                                         // original local coordinates
                                                                                         << "ly=" << ly
                                                                                         << "lz=" << lz
                                                                                         << "lx=" << x
                                                                                         // corrected local coordinated
                                                                                         << "lxT=" << lxT
                                                                                         << "lyT=" << lyT
                                                                                         << "lzT=" << lzT
                                                                                         // global uncorrected coordinates
                                                                                         << "gx=" << gx
                                                                                         << "gy=" << gy
                                                                                         << "gz=" << gz
                                                                                         // some transformations which are applied
                                                                                         << "invYZtoX=" << invYZtoX
                                                                                         << "YZtoNominalY=" << YZtoNominalY
                                                                                         << "YZtoNominalZ=" << YZtoNominalZ
                                                                                         << "\n";
    })
    x += dx;
    u += du;
    v += dv;
  }
}

GPUdi() void TPCFastTransformPOD::TransformIdealZ(int slice, float time, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms time TPC coordinates to local Z withing a slice
  /// Ideal transformation: only Vdrift from DCS.
  /// No space charge corrections, no time of flight correction
  ///

  float v = (time - mT0 - vertexTime) * mVdrift; // drift length cm
  getGeometry().convVtoLocal(slice, v, z);
}

GPUdi() void TPCFastTransformPOD::TransformIdeal(int slice, int row, float pad, float time, float& x, float& y, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a slice
  /// Ideal transformation: only Vdrift from DCS.
  /// No space charge corrections, no time of flight correction
  ///
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  x = rowInfo.x;
  float u = (pad - 0.5f * rowInfo.maxPad) * rowInfo.padWidth;
  float v = (time - mT0 - vertexTime) * mVdrift; // drift length cm
  getGeometry().convUVtoLocal(slice, u, v, y, z);
}

GPUdi() void TPCFastTransformPOD::convPadTimeToUV(int slice, int row, float pad, float time, float& u, float& v, float vertexTime) const
{
  bool sideC = (slice >= getGeometry().getNumberOfSlicesA());
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  const TPCFastTransformGeo::SliceInfo& sliceInfo = getGeometry().getSliceInfo(slice);
  float x = rowInfo.x;
  u = (pad - 0.5f * rowInfo.maxPad) * rowInfo.padWidth;
  float y = sideC ? -u : u; // pads are mirrorred on C-side
  float yLab = y * sliceInfo.cosAlpha + x * sliceInfo.sinAlpha;
  v = (time - mT0 - vertexTime) * (mVdrift + mVdriftCorrY * yLab) + mLdriftCorr; // drift length cm
}

GPUdi() void TPCFastTransformPOD::convTimeToVinTimeFrame(int slice, float time, float& v, float maxTimeBin) const
{
  v = (time - mT0 - maxTimeBin) * mVdrift + mLdriftCorr; // drift length cm
  if (slice < getGeometry().getNumberOfSlicesA()) {
    v += getGeometry().getTPCzLengthA();
  } else {
    v += getGeometry().getTPCzLengthC();
  }
}

GPUdi() void TPCFastTransformPOD::convPadTimeToUVinTimeFrame(int slice, int row, float pad, float time, float& u, float& v, float maxTimeBin) const
{
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  u = (pad - 0.5f * rowInfo.maxPad) * rowInfo.padWidth;
  convTimeToVinTimeFrame(slice, time, v, maxTimeBin);
}

GPUdi() float TPCFastTransformPOD::convZOffsetToVertexTime(int slice, float zOffset, float maxTimeBin) const
{
  if (slice < getGeometry().getNumberOfSlicesA()) {
    return maxTimeBin - (getGeometry().getTPCzLengthA() + zOffset) / mVdrift;
  } else {
    return maxTimeBin - (getGeometry().getTPCzLengthC() - zOffset) / mVdrift;
  }
}

GPUdi() float TPCFastTransformPOD::convVertexTimeToZOffset(int slice, float vertexTime, float maxTimeBin) const
{
  if (slice < getGeometry().getNumberOfSlicesA()) {
    return (maxTimeBin - vertexTime) * mVdrift - getGeometry().getTPCzLengthA();
  } else {
    return -((maxTimeBin - vertexTime) * mVdrift - getGeometry().getTPCzLengthC());
  }
}

GPUdi() void TPCFastTransformPOD::convUVtoPadTime(int slice, int row, float u, float v, float& pad, float& time, float vertexTime) const
{
  bool sideC = (slice >= getGeometry().getNumberOfSlicesA());

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  const TPCFastTransformGeo::SliceInfo& sliceInfo = getGeometry().getSliceInfo(slice);

  pad = u / rowInfo.padWidth + 0.5f * rowInfo.maxPad;

  float x = rowInfo.x;
  float y = sideC ? -u : u; // pads are mirrorred on C-side
  float yLab = y * sliceInfo.cosAlpha + x * sliceInfo.sinAlpha;
  time = mT0 + vertexTime + (v - mLdriftCorr) / (mVdrift + mVdriftCorrY * yLab);
}

GPUdi() void TPCFastTransformPOD::convVtoTime(float v, float& time, float vertexTime) const
{
  float yLab = 0.f;
  time = mT0 + vertexTime + (v - mLdriftCorr) / (mVdrift + mVdriftCorrY * yLab);
}

GPUdi() void TPCFastTransformPOD::convUVtoPadTimeInTimeFrame(int slice, int row, float u, float v, float& pad, float& time, float maxTimeBin) const
{
  if (slice < getGeometry().getNumberOfSlicesA()) {
    v -= getGeometry().getTPCzLengthA();
  } else {
    v -= getGeometry().getTPCzLengthC();
  }
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  pad = u / rowInfo.padWidth + 0.5f * rowInfo.maxPad;
  time = mT0 + maxTimeBin + (v - mLdriftCorr) / mVdrift;
}

GPUdi() void TPCFastTransformPOD::getTOFcorrection(int slice, int /*row*/, float x, float y, float z, float& dz) const
{
  // calculate time of flight correction for  z coordinate

  bool sideC = (slice >= getGeometry().getNumberOfSlicesA());
  float distZ = z - mPrimVtxZ;
  float dv = -GPUCommonMath::Sqrt(x * x + y * y + distZ * distZ) * mTOFcorr;
  dz = sideC ? dv : -dv;
}

GPUdi() float TPCFastTransformPOD::getMaxDriftTime(int slice, int row, float pad) const
{
  /// maximal possible drift time of the active area
  float maxL = getMaxDriftLength(slice, row, pad);
  bool sideC = (slice >= getGeometry().getNumberOfSlicesA());
  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  const TPCFastTransformGeo::SliceInfo& sliceInfo = getGeometry().getSliceInfo(slice);
  float x = rowInfo.x;
  float u = (pad - 0.5f * rowInfo.maxPad) * rowInfo.padWidth;

  float y = sideC ? -u : u; // pads are mirrorred on C-side
  float yLab = y * sliceInfo.cosAlpha + x * sliceInfo.sinAlpha;
  return mT0 + (maxL - mLdriftCorr) / (mVdrift + mVdriftCorrY * yLab);
}

GPUdi() float TPCFastTransformPOD::getMaxDriftTime(int slice, int row) const
{
  /// maximal possible drift time of the active area
  float maxL = getMaxDriftLength(slice, row);
  float maxTime = 0.f;
  convVtoTime(maxL, maxTime, 0.f);
  return maxTime;
}

GPUdi() float TPCFastTransformPOD::getMaxDriftTime(int slice) const
{
  /// maximal possible drift time of the active area
  float maxL = getMaxDriftLength(slice);
  float maxTime = 0.f;
  convVtoTime(maxL, maxTime, 0.f);
  return maxTime;
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // ALICEO2_GPU_TPCFastTransformPOD_H
