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

#include "GPUCommonRtypes.h"
#include "TPCFastTransform.h"
#ifndef GPUCA_GPUCODE
#include <memory>
#include <cstdlib>
#include "GPUCommonAlignedAlloc.h"
#endif

/*
Binary buffer should be cast to TPCFastTransformPOD class using static TPCFastTransformPOD& t = get(buffer); method,
so that its head becomes `this` pointer of the object.

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

namespace o2
{
namespace gpu
{
class TPCFastTransformPOD
{
 public:
  using SliceInfo = TPCFastSpaceChargeCorrection::SliceInfo; // obsolete
  using GridInfo = TPCFastSpaceChargeCorrection::GridInfo;
  using SectorRowInfo = TPCFastSpaceChargeCorrection::SectorRowInfo;

  using SplineTypeXYZ = TPCFastSpaceChargeCorrection::SplineTypeXYZ;
  using SplineTypeInvX = TPCFastSpaceChargeCorrection::SplineTypeInvX;
  using SplineTypeInvYZ = TPCFastSpaceChargeCorrection::SplineTypeInvYZ;
  using SplineType = TPCFastSpaceChargeCorrection::SplineType;

  /// convert prefilled buffer to TPCFastTransformPOD
  GPUd() static const TPCFastTransformPOD& get(const char* head) { return *reinterpret_cast<const TPCFastTransformPOD*>(head); }

  /// _______________ high level methods a la TPCFastTransform  _______________________
  ///
  // Methods taking extra reference transform are legacy compound transforms used to scale corrections.
  GPUd() void Transform(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime = 0) const;
  GPUd() void TransformXYZ(int32_t sector, int32_t row, float& x, float& y, float& z) const;

  /// Transformation in the time frame
  GPUd() void TransformInTimeFrame(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const;
  GPUd() void TransformInTimeFrame(int32_t sector, float time, float& z, float maxTimeBin) const;

  /// Inverse transformation
  GPUd() void InverseTransformInTimeFrame(int32_t sector, int32_t row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const;
  GPUd() float InverseTransformInTimeFrame(int32_t sector, float z, float maxTimeBin) const;

  /// Inverse transformation: Transformed Y and Z -> transformed X
  GPUd() void InverseTransformYZtoX(int32_t sector, int32_t row, float y, float z, float& x) const;

  /// Inverse transformation: Transformed Y and Z -> Y and Z, transformed w/o space charge correction
  GPUd() void InverseTransformYZtoNominalYZ(int32_t sector, int32_t row, float y, float z, float& ny, float& nz) const;

  /// Inverse transformation: Transformed X, Y and Z -> X, Y and Z, transformed w/o space charge correction
  GPUd() void InverseTransformXYZtoNominalXYZ(int32_t sector, int32_t row, float x, float y, float z, float& nx, float& ny, float& nz) const;

  /// Ideal transformation with Vdrift only - without calibration
  GPUd() void TransformIdeal(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime) const;
  GPUd() void TransformIdealZ(int32_t sector, float time, float& z, float vertexTime) const;

  GPUd() void convPadTimeToLocal(int32_t sector, int32_t row, float pad, float time, float& y, float& z, float vertexTime) const;
  GPUd() void convPadTimeToLocalInTimeFrame(int32_t sector, int32_t row, float pad, float time, float& y, float& z, float maxTimeBin) const;

  GPUd() void convLocalToPadTime(int32_t sector, int32_t row, float y, float z, float& pad, float& time, float vertexTime) const;
  GPUd() void convLocalToPadTimeInTimeFrame(int32_t sector, int32_t row, float y, float z, float& pad, float& time, float maxTimeBin) const;

  GPUd() float convTimeToZinTimeFrame(int32_t sector, float time, float maxTimeBin) const;
  GPUd() float convZtoTimeInTimeFrame(int32_t sector, float z, float maxTimeBin) const;
  GPUd() float convDeltaTimeToDeltaZinTimeFrame(int32_t sector, float deltaTime) const;
  GPUd() float convDeltaZtoDeltaTimeInTimeFrame(int32_t sector, float deltaZ) const;
  GPUd() float convDeltaZtoDeltaTimeInTimeFrameAbs(float deltaZ) const;
  GPUd() float convZOffsetToVertexTime(int32_t sector, float zOffset, float maxTimeBin) const;
  GPUd() float convVertexTimeToZOffset(int32_t sector, float vertexTime, float maxTimeBin) const;

  /// _______________ methods a la TPCFastSpaceChargeCorrection: cluster correction  _______________________
  void setApplyCorrectionOn() { mApplyCorrection = 1; }
  void setApplyCorrectionOff() { mApplyCorrection = 0; }
  bool isCorrectionApplied() { return mApplyCorrection; }

  /// TPC geometry information
  GPUd() const TPCFastTransformGeo& getGeometry() const { return mGeo; }

  /// Gives TPC sector & row info
  GPUd() const SectorRowInfo& getSectorRowInfo(int32_t sector, int32_t row) const { return mSectorRowInfos[NROWS * sector + row]; }

  /// Gives TPC sector & row info
  GPUd() SectorRowInfo& getSectorRowInfo(int32_t sector, int32_t row) { return mSectorRowInfos[NROWS * sector + row]; }

  /// Gives its own size including dynamic part
  GPUd() size_t size() const { return mTotalSize; }

  /// Gives the time stamp of the current calibaration parameters
  GPUd() long int getTimeStamp() const { return mTimeStamp; }

  /// Return mVDrift in cm / time bin
  GPUd() float getVDrift() const { return mVdrift; }

  /// Return T0 in time bin units
  GPUd() float getT0() const { return mT0; }

  /// Return IDC estimator
  GPUd() float getIDC() const { return mIDC; }

  /// Return Lumi estimator
  GPUd() float getLumi() const { return mLumi; }

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int32_t sector, int32_t row, float pad) const;

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int32_t sector, int32_t row) const;

  /// maximal possible drift time of the active area
  GPUd() float getMaxDriftTime(int32_t sector) const;

  /// Sets the time stamp of the current calibaration
  GPUd() void setTimeStamp(long int v) { mTimeStamp = v; }

  /// Sets current vdrift
  GPUd() void setVDrift(float v) { mVdrift = v; }

  /// Sets current T0
  GPUd() void setT0(float v) { mT0 = v; }

  /// Sets IDC estimator
  GPUd() void setIDC(float v) { mIDC = v; }

  /// Sets CTP Lumi estimator
  GPUd() void setLumi(float v) { mLumi = v; }

  GPUd() void setCalibration1(int64_t timeStamp, float t0, float vDrift);

  /// Gives a reference to a spline
  GPUd() const SplineType& getSpline(int32_t sector, int32_t row) const { return *reinterpret_cast<const SplineType*>(getThis() + getScenarioOffset(getSectorRowInfo(sector, row).splineScenarioID)); }

  /// Gives pointer to spline data
  GPUd() const float* getCorrectionData(int32_t sector, int32_t row, int32_t iSpline = 0) const { return reinterpret_cast<const float*>(getThis() + mSplineDataOffsets[sector][iSpline] + getSectorRowInfo(sector, row).dataOffsetBytes[iSpline]); }

  /// Gives const pointer to a spline for the inverse X correction
  GPUd() const SplineTypeInvX& getSplineInvX(int32_t sector, int32_t row) const { return reinterpret_cast<const SplineTypeInvX&>(getSpline(sector, row)); }

  /// Gives pointer to spline data for the inverse X correction
  GPUd() const float* getCorrectionDataInvX(int32_t sector, int32_t row) const { return getCorrectionData(sector, row, 1); }

  /// Gives const pointer to a spline for the inverse YZ correction
  GPUd() const SplineTypeInvYZ& getSplineInvYZ(int32_t sector, int32_t row) const { return reinterpret_cast<const SplineTypeInvYZ&>(getSpline(sector, row)); }

  /// Gives pointer to spline data for the inverse YZ correction
  GPUd() const float* getCorrectionDataInvYZ(int32_t sector, int32_t row) const { return getCorrectionData(sector, row, 2); }

  /// _______________ The main method: cluster correction  _______________________
  GPUdi() void getCorrectionLocal(int32_t sector, int32_t row, float y, float z, float& dx, float& dy, float& dz) const;

  /// inverse correction: Real Y and Z -> Real X
  GPUd() float getCorrectionXatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const;

  /// inverse correction: Real Y and Z -> measred Y and Z
  GPUd() void getCorrectionYZatRealYZ(int32_t sector, int32_t row, float realY, float realZ, float& measuredY, float& measuredZ) const;

  /// transformation in the sector local frame
  GPUd() void TransformLocal(int32_t sector, int32_t row, float& x, float& y, float& z) const;

  /// _______________  Utilities  _______________________________________________

  /// convert local y, z to internal grid coordinates u,v
  /// return values: u, v, scaling factor
  GPUd() void convLocalToGrid(int32_t sector, int32_t row, float y, float z, float& u, float& v, float& s) const;

  /// convert internal grid coordinates u,v to local y, z
  /// return values: y, z, scaling factor
  GPUd() void convGridToLocal(int32_t sector, int32_t row, float u, float v, float& y, float& z) const;

  /// convert real Y, Z to the internal grid coordinates
  /// return values: u, v, scaling factor
  GPUd() void convRealLocalToGrid(int32_t sector, int32_t row, float y, float z, float& u, float& v, float& s) const;

  /// convert internal grid coordinates to the real Y, Z
  /// return values: y, z
  GPUd() void convGridToRealLocal(int32_t sector, int32_t row, float u, float v, float& y, float& z) const;

  GPUd() bool isLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const;
  GPUd() bool isRealLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const;

#if !defined(GPUCA_GPUCODE)
  /// Create POD transform from old flat-buffer one. Provided vector will serve as a buffer
  static TPCFastTransformPOD* create(aligned_unique_buffer_ptr<TPCFastTransformPOD>& destVector, const TPCFastTransform& src);

  /// create filling only part corresponding to TPCFastSpaceChargeCorrection. Data members coming from TPCFastTransform (e.g. VDrift, T0..) are not set
  static TPCFastTransformPOD* create(aligned_unique_buffer_ptr<TPCFastTransformPOD>& destVector, const TPCFastSpaceChargeCorrection& src);

  static TPCFastTransformPOD* create(aligned_unique_buffer_ptr<TPCFastTransformPOD>& destVector, const TPCFastTransformPOD& src)
  {
    destVector.alloc(src.size());
    std::memcpy(destVector.get(), &src, src.size());
    return destVector.get();
  }

  static TPCFastTransformPOD* create(char* buff, size_t buffSize, const TPCFastTransform& src);
  static TPCFastTransformPOD* create(char* buff, size_t buffSize, const TPCFastSpaceChargeCorrection& src);
  static size_t estimateSize(const TPCFastTransform& src) { return estimateSize(src.getCorrection()); }
  static size_t estimateSize(const TPCFastSpaceChargeCorrection& origCorr);

  bool test(const TPCFastTransform& src, int32_t npoints = 100000) const { return test(src.getCorrection(), npoints); }
  bool test(const TPCFastSpaceChargeCorrection& origCorr, int32_t npoints = 100000) const;
#endif

  /// Print method
  void print() const;

  GPUd() float convDriftLengthToTime(float driftLength, float vertexTime) const;

  static constexpr int NROWS = 152;
  static constexpr int NSECTORS = TPCFastTransformGeo::getNumberOfSectors();
  static constexpr int NSplineIDs = 3; ///< number of spline data sets for each sector/row

 private:
#if !defined(GPUCA_GPUCODE)
  static constexpr size_t AlignmentBytes = 8;
  static size_t alignOffset(size_t offs)
  {
    auto res = offs % AlignmentBytes;
    return res ? offs + (AlignmentBytes - res) : offs;
  }
  GPUd() static TPCFastTransformPOD& getNonConst(char* head) { return *reinterpret_cast<TPCFastTransformPOD*>(head); }
#endif

  ///< get address to which the offset in bytes must be added to arrive to particular dynamic part
  GPUd() const char* getThis() const { return reinterpret_cast<const char*>(this); }

  ///< return offset of the spline object start (equivalent of mScenarioPtr in the TPCFastSpaceChargeCorrection)
  GPUd() size_t getScenarioOffset(int s) const { return (reinterpret_cast<const size_t*>(getThis() + mOffsScenariosOffsets))[s]; }

  GPUd() size_t getFlatBufferOffset(int s) const { return (reinterpret_cast<const size_t*>(getThis() + mOffsFlatBufferOffsets))[s]; }

  // Returns a pointer to the flat buffer of scenario isc, using only the
  // stored offset array (mOffsFlatBufferOffsets). No stale pointer involved.
  GPUd() const char* getSplineFlatBuffer(int32_t isc) const
  {
    const size_t* offs = reinterpret_cast<const size_t*>(getThis() + mOffsFlatBufferOffsets);
    return getThis() + offs[isc];
  }

  // Returns a pointer to mGridX2's flat buffer inside the spline flat buffer.
  // Reproduces the layout from Spline2DContainer::setActualBufferAddress using
  // only safe values: getFlatBufferSize() reads mNumberOfKnots/mUmax (plain ints).
  template <typename SplineT>
  GPUd() const char* getGridX2FlatBuffer(const SplineT& spline, int32_t isc) const
  {
    const size_t g1sz = spline.getGridX1().getFlatBufferSize();
    const size_t g2align = spline.getGridX2().getBufferAlignmentBytes();
    return getSplineFlatBuffer(isc) + FlatObject::alignSize(g1sz, g2align);
  }

  bool mApplyCorrection{};                                                          ///< flag to apply corrections
  int mNumberOfScenarios{};                                                         ///< Number of approximation spline scenarios
  size_t mTotalSize{};                                                              ///< total size of the buffer
  size_t mOffsScenariosOffsets{};                                                   ///< start of the array of mNumberOfScenarios offsets for each type of spline
  size_t mOffsFlatBufferOffsets{};                                                  ///< offset to array of mNumberOfScenarios flat buffer offsets
  size_t mSplineDataOffsets[TPCFastTransformGeo::getNumberOfSectors()][NSplineIDs]; ///< start of data for each sector and iSpline data
  long int mTimeStamp{};                                                            ///< time stamp of the current calibration
  float mT0;                                                                        ///< T0 in [time bin]
  float mVdrift;                                                                    ///< VDrift in  [cm/time bin]
  float mLumi;                                                                      ///< luminosity estimator (for info only)
  float mIDC;                                                                       ///< IDC estimator (for info only)

  TPCFastTransformGeo mGeo; ///< TPC geometry information
  SectorRowInfo mSectorRowInfos[NROWS * TPCFastTransformGeo::getNumberOfSectors()];

  ClassDefNV(TPCFastTransformPOD, 0);
};

GPUdi() void TPCFastTransformPOD::getCorrectionLocal(int32_t sector, int32_t row, float y, float z, float& dx, float& dy, float& dz) const
{
  const auto& info = getSectorRowInfo(sector, row);
  const int32_t isc = info.splineScenarioID;
  const SplineType& spline = getSpline(sector, row);
  const float* splineData = getCorrectionData(sector, row);

  float u, v, s;
  convLocalToGrid(sector, row, y, z, u, v, s);

  const char* g1buf = getSplineFlatBuffer(isc);
  const char* g2buf = getGridX2FlatBuffer(spline, isc);

  float dxyz[3];
  spline.interpolateAtUZeroCopy(g1buf, g2buf, splineData, u, v, dxyz);

  if (CAMath::Abs(dxyz[0]) > 100.f || CAMath::Abs(dxyz[1]) > 100.f || CAMath::Abs(dxyz[2]) > 100.f) {
    s = 0.f; // TODO: DR: Protect from FPEs, fix upstream and remove once guaranteed that it is fixed
  }

  dx = s * GPUCommonMath::Clamp(dxyz[0], info.minCorr[0], info.maxCorr[0]);
  dy = s * GPUCommonMath::Clamp(dxyz[1], info.minCorr[1], info.maxCorr[1]);
  dz = s * GPUCommonMath::Clamp(dxyz[2], info.minCorr[2], info.maxCorr[2]);
}

GPUdi() float TPCFastTransformPOD::getCorrectionXatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const
{
  const auto& info = getSectorRowInfo(sector, row);
  float u, v, s;
  convRealLocalToGrid(sector, row, realY, realZ, u, v, s);

  const int32_t isc = info.splineScenarioID;
  const auto& spline = getSplineInvX(sector, row);
  const char* g1buf = getSplineFlatBuffer(isc);
  const char* g2buf = getGridX2FlatBuffer(spline, isc);

  float dx = 0;
  spline.interpolateAtUZeroCopy(g1buf, g2buf, getCorrectionDataInvX(sector, row), u, v, &dx);
  if (CAMath::Abs(dx) > 100.f) {
    s = 0.f; // TODO: DR: Protect from FPEs, fix upstream and remove once guaranteed that it is fixed
  }
  dx = s * GPUCommonMath::Clamp(dx, info.minCorr[0], info.maxCorr[0]);
  return dx;
}

GPUdi() void TPCFastTransformPOD::getCorrectionYZatRealYZ(int32_t sector, int32_t row, float realY, float realZ, float& y, float& z) const
{
  float u, v, s;
  convRealLocalToGrid(sector, row, realY, realZ, u, v, s);
  const auto& info = getSectorRowInfo(sector, row);
  const int32_t isc = info.splineScenarioID;
  const auto& spline = getSplineInvYZ(sector, row);
  const char* g1buf = getSplineFlatBuffer(isc);
  const char* g2buf = getGridX2FlatBuffer(spline, isc);

  float dyz[2];
  spline.interpolateAtUZeroCopy(g1buf, g2buf, getCorrectionDataInvYZ(sector, row), u, v, dyz);
  if (CAMath::Abs(dyz[0]) > 100.f || CAMath::Abs(dyz[1]) > 100.f) {
    s = 0.f; // TODO: DR: Protect from FPEs, fix upstream and remove once guaranteed that it is fixed
  }
  y = s * GPUCommonMath::Clamp(dyz[0], info.minCorr[1], info.maxCorr[1]);
  z = s * GPUCommonMath::Clamp(dyz[1], info.minCorr[2], info.maxCorr[2]);
}

GPUdi() void TPCFastTransformPOD::convLocalToGrid(int32_t sector, int32_t row, float y, float z, float& u, float& v, float& s) const
{
  /// convert local y, z to internal grid coordinates u,v
  /// return values: u, v, scaling factor
  const SplineType& spline = getSpline(sector, row);
  getSectorRowInfo(sector, row).gridMeasured.convLocalToGridUntruncated(y, z, u, v, s);
  // shrink to the grid
  u = GPUCommonMath::Clamp(u, 0.f, (float)spline.getGridX1().getUmax());
  v = GPUCommonMath::Clamp(v, 0.f, (float)spline.getGridX2().getUmax());
}

GPUdi() void TPCFastTransformPOD::convGridToLocal(int32_t sector, int32_t row, float gridU, float gridV, float& y, float& z) const
{
  /// convert internal grid coordinates u,v to local y, z
  getSectorRowInfo(sector, row).gridMeasured.convGridToLocal(gridU, gridV, y, z);
}

GPUdi() void TPCFastTransformPOD::convRealLocalToGrid(int32_t sector, int32_t row, float y, float z, float& u, float& v, float& s) const
{
  /// convert real y, z to the internal grid coordinates + scale
  const SplineType& spline = getSpline(sector, row);
  getSectorRowInfo(sector, row).gridReal.convLocalToGridUntruncated(y, z, u, v, s);
  // shrink to the grid
  u = GPUCommonMath::Clamp(u, 0.f, (float)spline.getGridX1().getUmax());
  v = GPUCommonMath::Clamp(v, 0.f, (float)spline.getGridX2().getUmax());
}

GPUdi() void TPCFastTransformPOD::convGridToRealLocal(int32_t sector, int32_t row, float gridU, float gridV, float& y, float& z) const
{
  /// convert internal grid coordinates u,v to the real y, z
  getSectorRowInfo(sector, row).gridReal.convGridToLocal(gridU, gridV, y, z);
}

GPUdi() bool TPCFastTransformPOD::isLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// check if local y, z are inside the grid
  float u, v, s;
  getSectorRowInfo(sector, row).gridMeasured.convLocalToGridUntruncated(y, z, u, v, s);
  const auto& spline = getSpline(sector, row);
  // shrink to the grid
  if (u < 0.f || u > (float)spline.getGridX1().getUmax() || //
      v < 0.f || v > (float)spline.getGridX2().getUmax()) {
    return false;
  }
  return true;
}

GPUdi() bool TPCFastTransformPOD::isRealLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// check if local y, z are inside the grid
  float u, v, s;
  getSectorRowInfo(sector, row).gridReal.convLocalToGridUntruncated(y, z, u, v, s);
  const auto& spline = getSpline(sector, row);
  // shrink to the grid
  if (u < 0.f || u > (float)spline.getGridX1().getUmax() || //
      v < 0.f || v > (float)spline.getGridX2().getUmax()) {
    return false;
  }
  return true;
}

GPUdi() void TPCFastTransformPOD::TransformLocal(int32_t sector, int32_t row, float& x, float& y, float& z) const
{
  if (!mApplyCorrection) {
    return;
  }
  float dx, dy, dz;
  getCorrectionLocal(sector, row, y, z, dx, dy, dz);

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    float lx = x, ly = y, lz = z;
    float gx, gy, gz;
    getGeometry().convLocalToGlobal(sector, lx, ly, lz, gx, gy, gz);
    float lxT = lx + dx;
    float lyT = ly + dy;
    float lzT = lz + dz;
    float invYZtoX;
    InverseTransformYZtoX(sector, row, lyT, lzT, invYZtoX);

    float YZtoNominalY;
    float YZtoNominalZ;
    InverseTransformYZtoNominalYZ(sector, row, lyT, lzT, YZtoNominalY, YZtoNominalZ);

    o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_Transform").data()
                                                                                       // corrections in x, u, v
                                                                                       << "dx=" << dx
                                                                                       << "dy=" << dy
                                                                                       << "dz=" << dz
                                                                                       << "row=" << row
                                                                                       << "sector=" << sector
                                                                                       // original local coordinates
                                                                                       << "ly=" << ly
                                                                                       << "lz=" << lz
                                                                                       << "lx=" << lx
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
  y += dy;
  z += dz;
}

GPUdi() void TPCFastTransformPOD::Transform(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a sector
  /// taking calibration into account.
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  x = rowInfo.x;
  convPadTimeToLocal(sector, row, pad, time, y, z, vertexTime);
  TransformLocal(sector, row, x, y, z);
}

GPUdi() void TPCFastTransformPOD::TransformXYZ(int32_t sector, int32_t row, float& x, float& y, float& z) const
{

  TransformLocal(sector, row, x, y, z);
}

GPUdi() void TPCFastTransformPOD::TransformInTimeFrame(int32_t sector, float time, float& z, float maxTimeBin) const
{
  float l = (time - mT0 - maxTimeBin) * mVdrift; // drift length cm
  z = getGeometry().convDriftLengthToZ1(sector, l);
}

GPUdi() void TPCFastTransformPOD::TransformInTimeFrame(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float maxTimeBin) const
{
  /// _______________ Special cluster transformation for a time frame _______________________
  ///
  /// Same as Transform(), but clusters are shifted in z such, that Z(maxTimeBin)==0
  /// Corrections and Time-Of-Flight correction are not alpplied.
  ///

  const TPCFastTransformGeo::RowInfo& rowInfo = getGeometry().getRowInfo(row);
  x = rowInfo.x;
  convPadTimeToLocalInTimeFrame(sector, row, pad, time, y, z, maxTimeBin);
}

GPUdi() void TPCFastTransformPOD::InverseTransformInTimeFrame(int32_t sector, int32_t row, float /*x*/, float y, float z, float& pad, float& time, float maxTimeBin) const
{
  /// Inverse transformation to TransformInTimeFrame
  convLocalToPadTimeInTimeFrame(sector, row, y, z, pad, time, maxTimeBin);
}

GPUdi() float TPCFastTransformPOD::InverseTransformInTimeFrame(int32_t sector, float z, float maxTimeBin) const
{
  float pad, time;
  InverseTransformInTimeFrame(sector, 0, 0, 0, z, pad, time, maxTimeBin);
  return time;
}

GPUdi() void TPCFastTransformPOD::TransformIdealZ(int32_t sector, float time, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms time TPC coordinates to local Z withing a sector
  /// Ideal transformation: only Vdrift from DCS.
  /// No space charge corrections, no time of flight correction
  ///

  float l = (time - mT0 - vertexTime) * mVdrift; // drift length cm
  z = getGeometry().convDriftLengthToZ1(sector, l);
}

GPUdi() void TPCFastTransformPOD::TransformIdeal(int32_t sector, int32_t row, float pad, float time, float& x, float& y, float& z, float vertexTime) const
{
  /// _______________ The main method: cluster transformation _______________________
  ///
  /// Transforms raw TPC coordinates to local XYZ withing a sector
  /// Ideal transformation: only Vdrift from DCS.
  /// No space charge corrections, no time of flight correction
  ///

  x = getGeometry().getRowInfo(row).x;
  float driftLength = (time - mT0 - vertexTime) * mVdrift; // drift length cm
  getGeometry().convPadDriftLengthToLocal(sector, row, pad, driftLength, y, z);
}

GPUdi() float TPCFastTransformPOD::convTimeToZinTimeFrame(int32_t sector, float time, float maxTimeBin) const
{
  /// _______________ Special cluster transformation for a time frame _______________________
  ///
  /// Same as Transform(), but clusters are shifted in z such, that Z(maxTimeBin)==0
  /// Corrections and Time-Of-Flight correction are not alpplied.
  /// Only Z coordinate.
  ///

  float v = (time - mT0 - maxTimeBin) * mVdrift; // drift length cm
  float z = (sector < getGeometry().getNumberOfSectorsA()) ? -v : v;
  return z;
}

GPUdi() float TPCFastTransformPOD::convZtoTimeInTimeFrame(int32_t sector, float z, float maxTimeBin) const
{
  /// Inverse transformation of convTimeToZinTimeFrame()
  float v = (sector < getGeometry().getNumberOfSectorsA()) ? -z : z;
  return mT0 + maxTimeBin + v / mVdrift;
}

GPUdi() float TPCFastTransformPOD::convDeltaTimeToDeltaZinTimeFrame(int32_t sector, float deltaTime) const
{
  float deltaZ = deltaTime * mVdrift;
  return sector < getGeometry().getNumberOfSectorsA() ? -deltaZ : deltaZ;
}

GPUdi() float TPCFastTransformPOD::convDeltaZtoDeltaTimeInTimeFrameAbs(float deltaZ) const
{
  return deltaZ / mVdrift;
}

GPUdi() float TPCFastTransformPOD::convDeltaZtoDeltaTimeInTimeFrame(int32_t sector, float deltaZ) const
{
  float deltaT = deltaZ / mVdrift;
  return sector < getGeometry().getNumberOfSectorsA() ? -deltaT : deltaT;
}

GPUdi() float TPCFastTransformPOD::getMaxDriftTime(int32_t sector, int32_t row, float pad) const
{
  /// maximal possible drift time of the active area
  return convDriftLengthToTime(getGeometry().getTPCzLength(), 0.f);
}

GPUdi() float TPCFastTransformPOD::getMaxDriftTime(int32_t sector, int32_t row) const
{
  /// maximal possible drift time of the active area
  return convDriftLengthToTime(getGeometry().getTPCzLength(), 0.f);
}

GPUdi() float TPCFastTransformPOD::getMaxDriftTime(int32_t sector) const
{
  /// maximal possible drift time of the active area
  return convDriftLengthToTime(getGeometry().getTPCzLength(), 0.f);
}

GPUdi() void TPCFastTransformPOD::InverseTransformYZtoX(int32_t sector, int32_t row, float realY, float realZ, float& realX) const
{
  /// Transformation y,z -> x
  float dx = 0.f;
  dx = getCorrectionXatRealYZ(sector, row, realY, realZ);
  realX = getGeometry().getRowInfo(row).x + dx;

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_InverseTransformYZtoX").data()
                                                                                       << "sector=" << sector
                                                                                       << "row=" << row
                                                                                       << "y=" << realY
                                                                                       << "z=" << realZ
                                                                                       << "x=" << realX
                                                                                       << "\n";
  })
}

GPUdi() void TPCFastTransformPOD::InverseTransformYZtoNominalYZ(int32_t sector, int32_t row, float realY, float realZ, float& measuredY, float& measuredZ) const
{
  /// Transformation real y,z -> measured y,z
  float dy, dz;
  getCorrectionYZatRealYZ(sector, row, realY, realZ, dy, dz);
  measuredY = realY - dy;
  measuredZ = realZ - dz;

  GPUCA_DEBUG_STREAMER_CHECK(if (o2::utils::DebugStreamer::checkStream(o2::utils::StreamFlags::streamFastTransform)) {
    o2::utils::DebugStreamer::instance()->getStreamer("debug_fasttransform", "UPDATE") << o2::utils::DebugStreamer::instance()->getUniqueTreeName("tree_InverseTransformYZtoNominalYZ").data()
                                                                                       << "sector=" << sector
                                                                                       << "row=" << row
                                                                                       << "real y=" << realY
                                                                                       << "real z=" << realZ
                                                                                       << "measured y=" << measuredY
                                                                                       << "measured z=" << measuredZ
                                                                                       << "\n";
  })
}

GPUdi() void TPCFastTransformPOD::convPadTimeToLocal(int32_t sector, int32_t row, float pad, float time, float& y, float& z, float vertexTime) const
{
  float l = (time - mT0 - vertexTime) * mVdrift;
  getGeometry().convPadDriftLengthToLocal(sector, row, pad, l, y, z);
}

GPUdi() void TPCFastTransformPOD::convPadTimeToLocalInTimeFrame(int32_t sector, int32_t row, float pad, float time, float& y, float& z, float maxTimeBin) const
{
  float l = getGeometry().getTPCzLength() + (time - mT0 - maxTimeBin) * mVdrift;
  getGeometry().convPadDriftLengthToLocal(sector, row, pad, l, y, z);
}

GPUdi() void TPCFastTransformPOD::convLocalToPadTimeInTimeFrame(int32_t sector, int32_t row, float y, float z, float& pad, float& time, float maxTimeBin) const
{
  float length = 0;
  getGeometry().convLocalToPadDriftLength(sector, row, y, z, pad, length);
  time = convDriftLengthToTime(length, maxTimeBin);
}

GPUdi() float TPCFastTransformPOD::convDriftLengthToTime(float driftLength, float vertexTime) const
{
  return (mT0 + vertexTime + driftLength / mVdrift);
}

GPUdi() float TPCFastTransformPOD::convZOffsetToVertexTime(int32_t sector, float zOffset, float maxTimeBin) const
{
  if (sector < getGeometry().getNumberOfSectorsA()) {
    return maxTimeBin - (getGeometry().getTPCzLength() + zOffset) / mVdrift;
  } else {
    return maxTimeBin - (getGeometry().getTPCzLength() - zOffset) / mVdrift;
  }
}

GPUdi() float TPCFastTransformPOD::convVertexTimeToZOffset(int32_t sector, float vertexTime, float maxTimeBin) const
{
  if (sector < getGeometry().getNumberOfSectorsA()) {
    return (maxTimeBin - vertexTime) * mVdrift - getGeometry().getTPCzLength();
  } else {
    return -((maxTimeBin - vertexTime) * mVdrift - getGeometry().getTPCzLength());
  }
}

#ifndef GPUCA_GPUCODE_DEVICE // Functions not needed during GPU processing
GPUdi() void TPCFastTransformPOD::setCalibration1(int64_t timeStamp, float t0, float vDrift)
{
  mTimeStamp = timeStamp;
  mT0 = t0;
  mVdrift = vDrift;
}

GPUdi() void TPCFastTransformPOD::InverseTransformXYZtoNominalXYZ(int32_t sector, int32_t row, float x, float y, float z, float& nx, float& ny, float& nz) const
{
  /// Inverse transformation: Transformed X, Y and Z -> X, Y and Z, transformed w/o space charge correction
  int32_t row2 = row + 1;
  if (row2 >= getGeometry().getNumberOfRows()) {
    row2 = row - 1;
  }
  float nx1, ny1, nz1; // nominal coordinates for row
  float nx2, ny2, nz2; // nominal coordinates for row2
  nx1 = getGeometry().getRowInfo(row).x;
  nx2 = getGeometry().getRowInfo(row2).x;
  InverseTransformYZtoNominalYZ(sector, row, y, z, ny1, nz1);
  InverseTransformYZtoNominalYZ(sector, row2, y, z, ny2, nz2);
  float c1 = (nx2 - nx) / (nx2 - nx1);
  float c2 = (nx - nx1) / (nx2 - nx1);
  nx = x;
  ny = (ny1 * c1 + ny2 * c2);
  nz = (nz1 * c1 + nz2 * c2);
}
#endif // GPUCA_GPUCODE_DEVICE

} // namespace gpu
} // namespace o2

#endif
