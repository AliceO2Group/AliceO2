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
#include "GPUCommonMath.h"

namespace o2
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
  // obsolete structure, declared here only for backward compatibility
  struct SliceInfo {
    ClassDefNV(SliceInfo, 2);
  };

  struct GridInfo {
   private:
    float y0{0.f};                 ///< Y coordinate of the U-grid start
    float yScale{0.f};             //< scale Y to U-grid coordinate
    float z0{0.f};                 ///< Z coordinate of the V-grid start
    float zScale{0.f};             //< scale Z to V-grid coordinate
    float zOut{0.f};               // outer z of the grid;
    float splineScalingWithZ{0.f}; ///< spline scaling factor in the Z region between the zOut and the readout plane

   public:
    void set(float y0_, float yScale_, float z0_, float zScale_, float zOut_, float zReadout_)
    {
      this->y0 = y0_;
      this->yScale = yScale_;
      this->z0 = z0_;
      this->zScale = zScale_;
      this->zOut = zOut_;
      // no scaling when the distance to the readout is too small
      this->splineScalingWithZ = fabs(zReadout_ - zOut_) > 1. ? 1. / (zReadout_ - zOut_) : 0.;
    }

    float getY0() const { return y0; }
    float getYscale() const { return yScale; }
    float getZ0() const { return z0; }
    float getZscale() const { return zScale; }

    GPUd() float getSpineScaleForZ(float z) const
    {
      return 1.f - GPUCommonMath::Clamp((z - zOut) * splineScalingWithZ, 0.f, 1.f);
    }

    /// convert local y, z to internal grid coordinates u,v, and spline scale
    GPUd() std::array<float, 3> convLocalToGridUntruncated(float y, float z) const
    {
      return {(y - y0) * yScale, (z - z0) * zScale, getSpineScaleForZ(z)};
    }

    /// convert internal grid coordinates u,v to local y, z
    std::array<float, 2> convGridToLocal(float gridU, float gridV) const
    {
      return {y0 + gridU / yScale, z0 + gridV / zScale};
    }
    ClassDefNV(GridInfo, 1);
  };

  struct SectorRowInfo {
    int32_t splineScenarioID{0};  ///< scenario index (which of Spline2D splines to use)
    size_t dataOffsetBytes[3]{0}; ///< offset for the spline data withing a TPC sector

    GridInfo gridMeasured; ///< grid info for measured coordinates
    GridInfo gridReal;     ///< grid info for real coordinates

    float minCorr[3]{-10.f, -10.f, -10.f}; ///< min correction for dX, dY, dZ
    float maxCorr[3]{10.f, 10.f, 10.f};    ///< max correction for dX, dY, dZ

    void resetMaxValues()
    {
      minCorr[0] = -1.f;
      maxCorr[0] = 1.f;
      minCorr[1] = -1.f;
      maxCorr[1] = 1.f;
      minCorr[2] = -1.f;
      maxCorr[2] = 1.f;
    }

    void updateMaxValues(float dx, float du, float dv)
    {
      minCorr[0] = GPUCommonMath::Min(minCorr[0], dx);
      maxCorr[0] = GPUCommonMath::Max(maxCorr[0], dx);

      minCorr[1] = GPUCommonMath::Min(minCorr[1], du);
      maxCorr[1] = GPUCommonMath::Max(maxCorr[1], du);

      minCorr[2] = GPUCommonMath::Min(minCorr[2], dv);
      maxCorr[2] = GPUCommonMath::Max(maxCorr[2], dv);
    }

    void updateMaxValues(std::array<float, 3> dxdudv, float scale)
    {
      float dx = dxdudv[0] * scale;
      float du = dxdudv[1] * scale;
      float dv = dxdudv[2] * scale;
      updateMaxValues(dx, du, dv);
    }

    std::array<float, 3> getMaxValues() const
    {
      return {maxCorr[0], maxCorr[1], maxCorr[2]};
    }

    std::array<float, 3> getMinValues() const
    {
      return {minCorr[0], minCorr[1], minCorr[2]};
    }

    ClassDefNV(SectorRowInfo, 2);
  };

  typedef Spline2D<float, 3> SplineTypeXYZ;
  typedef Spline2D<float, 1> SplineTypeInvX;
  typedef Spline2D<float, 2> SplineTypeInvYZ;

  typedef SplineTypeXYZ SplineType;

  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCFastSpaceChargeCorrection();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneTo[In/Ex]ternalBuffer() instead
  TPCFastSpaceChargeCorrection(const TPCFastSpaceChargeCorrection&) = delete;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneTo[In/Ex]ternalBuffer() instead
  TPCFastSpaceChargeCorrection& operator=(const TPCFastSpaceChargeCorrection&) = delete;

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

  void setActualBufferAddressOld(char* actualFlatBufferPtr);
  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

  /// _______________  Construction interface  ________________________

  /// Starts the construction procedure, reserves temporary memory
  void startConstruction(const TPCFastTransformGeo& geo, int32_t numberOfSplineScenarios);

  /// Initializes a TPC row
  void setRowScenarioID(int32_t iSector, int32_t iRow, int32_t iScenario);

  /// Sets approximation scenario
  void setSplineScenario(int32_t scenarioIndex, const SplineType& spline);

  /// Finishes construction: puts everything to the flat buffer, releases temporary memory
  void finishConstruction();

  void constructWithNoCorrection(const TPCFastTransformGeo& geo);

  /// _______________  Initialization interface  ________________________

  /// Set no correction
  GPUd() void setNoCorrection();

  /// Sets the time stamp of the current calibaration
  GPUd() void setTimeStamp(int64_t v) { mTimeStamp = v; }

  /// Gives const pointer to a spline
  GPUd() const SplineType& getSpline(int32_t sector, int32_t row) const;

  /// Gives pointer to a spline
  GPUd() SplineType& getSpline(int32_t sector, int32_t row);

  /// Gives pointer to spline data
  GPUd() float* getCorrectionData(int32_t sector, int32_t row, int32_t iSpline = 0);

  /// Gives pointer to spline data
  GPUd() const float* getCorrectionData(int32_t sector, int32_t row, int32_t iSpline = 0) const;

  /// Gives const pointer to a spline for the inverse X correction
  GPUd() const SplineTypeInvX& getSplineInvX(int32_t sector, int32_t row) const;

  /// Gives pointer to a spline for the inverse X correction
  GPUd() SplineTypeInvX& getSplineInvX(int32_t sector, int32_t row);

  /// Gives pointer to spline data for the inverse X correction
  GPUd() float* getCorrectionDataInvX(int32_t sector, int32_t row);

  /// Gives pointer to spline data for the inverse X correction
  GPUd() const float* getCorrectionDataInvX(int32_t sector, int32_t row) const;

  /// Gives const pointer to a spline for the inverse YZ correction
  GPUd() const SplineTypeInvYZ& getSplineInvYZ(int32_t sector, int32_t row) const;

  /// Gives pointer to a spline for the inverse YZ correction
  GPUd() SplineTypeInvYZ& getSplineInvYZ(int32_t sector, int32_t row);

  /// Gives pointer to spline data for the inverse YZ correction
  GPUd() float* getCorrectionDataInvYZ(int32_t sector, int32_t row);

  /// Gives pointer to spline data for the inverse YZ correction
  GPUd() const float* getCorrectionDataInvYZ(int32_t sector, int32_t row) const;

  /// _______________ The main method: cluster correction  _______________________
  ///
  // GPUd() int32_t getCorrectionInternal(int32_t sector, int32_t row, float u, float v, float& dx, float& du, float& dv) const;

  GPUdi() std::array<float, 3> getCorrectionLocal(int32_t sector, int32_t row, float y, float z) const;

  /// inverse correction: Real Y and Z -> Real X
  GPUd() float getCorrectionXatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const;

  /// inverse correction: Real Y and Z -> measred Y and Z
  GPUd() std::array<float, 2> getCorrectionYZatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const;

  /// _______________  Utilities  _______________________________________________

  /// convert local y, z to internal grid coordinates u,v
  /// return values: u, v, scaling factor
  GPUd() std::array<float, 3> convLocalToGrid(int32_t sector, int32_t row, float y, float z) const;

  /// convert internal grid coordinates u,v to local y, z
  /// return values: y, z, scaling factor
  GPUd() std::array<float, 2> convGridToLocal(int32_t sector, int32_t row, float u, float v) const;

  /// convert real Y, Z to the internal grid coordinates
  /// return values: u, v, scaling factor
  GPUd() std::array<float, 3> convRealLocalToGrid(int32_t sector, int32_t row, float y, float z) const;

  /// convert internal grid coordinates to the real Y, Z
  /// return values: y, z
  GPUd() std::array<float, 2> convGridToRealLocal(int32_t sector, int32_t row, float u, float v) const;

  GPUd() bool isLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const;
  GPUd() bool isRealLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const;

  /// TPC geometry information
  GPUd() const TPCFastTransformGeo& getGeometry() const
  {
    return mGeo;
  }

  /// Gives the time stamp of the current calibaration parameters
  int64_t getTimeStamp() const { return mTimeStamp; }

  /// Gives TPC sector & row info
  GPUd() const SectorRowInfo& getSectorRowInfo(int32_t sector, int32_t row) const
  {
    return mSectorRowInfos[mGeo.getMaxNumberOfRows() * sector + row];
  }

  /// Gives TPC sector & row info
  GPUd() SectorRowInfo& getSectorRowInfo(int32_t sector, int32_t row)
  {
    return mSectorRowInfos[mGeo.getMaxNumberOfRows() * sector + row];
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

  /// _______________  Data members  _______________________________________________

  /// _______________  Construction control  _______________________________________________

  SplineType* mConstructionScenarios = nullptr; //! (transient!!) Temporary container for spline scenarios

  /// _______________  Geometry  _______________________________________________

  TPCFastTransformGeo mGeo; ///< TPC geometry information

  int32_t mNumberOfScenarios; ///< Number of approximation spline scenarios

  SplineType* mScenarioPtr; //! (transient!!) pointer to spline scenarios

  /// _______________  Calibration data  _______________________________________________

  int64_t mTimeStamp; ///< time stamp of the current calibration

  char* mCorrectionData[3]; //! (transient!!) pointer to the spline data in the flat buffer

  size_t mCorrectionDataSize[3]; ///< size of the data per transformation (direct, inverseX, inverse YZ) in the flat buffer

  /// Class version. It is used to read older versions from disc.
  /// The default version 3 is the one before this field was introduced.
  /// The actual version must be set in startConstruction().
  int32_t mClassVersion{3};

  SectorRowInfo mSectorRowInfos[TPCFastTransformGeo::getNumberOfSectors() * TPCFastTransformGeo::getMaxNumberOfRows()]; ///< SectorRowInfo array

  ClassDefNV(TPCFastSpaceChargeCorrection, 4);
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

GPUdi() const TPCFastSpaceChargeCorrection::SplineType& TPCFastSpaceChargeCorrection::getSpline(int32_t sector, int32_t row) const
{
  /// Gives const pointer to spline
  return mScenarioPtr[getSectorRowInfo(sector, row).splineScenarioID];
}

GPUdi() TPCFastSpaceChargeCorrection::SplineType& TPCFastSpaceChargeCorrection::getSpline(int32_t sector, int32_t row)
{
  /// Gives pointer to spline
  return mScenarioPtr[getSectorRowInfo(sector, row).splineScenarioID];
}

GPUdi() float* TPCFastSpaceChargeCorrection::getCorrectionData(int32_t sector, int32_t row, int32_t iSpline)
{
  /// Gives pointer to spline data
  return reinterpret_cast<float*>(mCorrectionData[iSpline] + getSectorRowInfo(sector, row).dataOffsetBytes[iSpline]);
}

GPUdi() const float* TPCFastSpaceChargeCorrection::getCorrectionData(int32_t sector, int32_t row, int32_t iSpline) const
{
  /// Gives pointer to spline data
  return reinterpret_cast<const float*>(mCorrectionData[iSpline] + getSectorRowInfo(sector, row).dataOffsetBytes[iSpline]);
}

GPUdi() TPCFastSpaceChargeCorrection::SplineTypeInvX& TPCFastSpaceChargeCorrection::getSplineInvX(int32_t sector, int32_t row)
{
  /// Gives pointer to spline for the inverse X correction
  return reinterpret_cast<SplineTypeInvX&>(getSpline(sector, row));
}

GPUdi() const TPCFastSpaceChargeCorrection::SplineTypeInvX& TPCFastSpaceChargeCorrection::getSplineInvX(int32_t sector, int32_t row) const
{
  /// Gives const pointer to spline for the inverse X correction
  return reinterpret_cast<const SplineTypeInvX&>(getSpline(sector, row));
}

GPUdi() float* TPCFastSpaceChargeCorrection::getCorrectionDataInvX(int32_t sector, int32_t row)
{
  /// Gives pointer to spline data for the inverse X correction
  return getCorrectionData(sector, row, 1);
}

GPUdi() const float* TPCFastSpaceChargeCorrection::getCorrectionDataInvX(int32_t sector, int32_t row) const
{
  /// Gives pointer to spline data for the inverse X correction
  return getCorrectionData(sector, row, 1);
}

GPUdi() TPCFastSpaceChargeCorrection::SplineTypeInvYZ& TPCFastSpaceChargeCorrection::getSplineInvYZ(int32_t sector, int32_t row)
{
  /// Gives pointer to spline for the inverse YZ correction
  return reinterpret_cast<SplineTypeInvYZ&>(getSpline(sector, row));
}

GPUdi() const TPCFastSpaceChargeCorrection::SplineTypeInvYZ& TPCFastSpaceChargeCorrection::getSplineInvYZ(int32_t sector, int32_t row) const
{
  /// Gives const pointer to spline for the inverse YZ correction
  return reinterpret_cast<const SplineTypeInvYZ&>(getSpline(sector, row));
}

GPUdi() float* TPCFastSpaceChargeCorrection::getCorrectionDataInvYZ(int32_t sector, int32_t row)
{
  /// Gives pointer to spline data for the inverse YZ correction
  return getCorrectionData(sector, row, 2);
}

GPUdi() const float* TPCFastSpaceChargeCorrection::getCorrectionDataInvYZ(int32_t sector, int32_t row) const
{
  /// Gives pointer to spline data for the inverse YZ correction
  return getCorrectionData(sector, row, 2);
}

GPUdi() std::array<float, 3> TPCFastSpaceChargeCorrection::convLocalToGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// convert local y, z to internal grid coordinates u,v
  /// return values: u, v, scaling factor
  const SplineType& spline = getSpline(sector, row);
  auto val = getSectorRowInfo(sector, row).gridMeasured.convLocalToGridUntruncated(y, z);
  // shrink to the grid
  val[0] = GPUCommonMath::Clamp(val[0], 0.f, (float)spline.getGridX1().getUmax());
  val[1] = GPUCommonMath::Clamp(val[1], 0.f, (float)spline.getGridX2().getUmax());
  return val;
}

GPUdi() bool TPCFastSpaceChargeCorrection::isLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// check if local y, z are inside the grid
  auto val = getSectorRowInfo(sector, row).gridMeasured.convLocalToGridUntruncated(y, z);
  const auto& spline = getSpline(sector, row);
  // shrink to the grid
  if (val[0] < 0.f || val[0] > (float)spline.getGridX1().getUmax() || //
      val[1] < 0.f || val[1] > (float)spline.getGridX2().getUmax()) {
    return false;
  }
  return true;
}

GPUdi() bool TPCFastSpaceChargeCorrection::isRealLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// check if local y, z are inside the grid
  auto val = getSectorRowInfo(sector, row).gridReal.convLocalToGridUntruncated(y, z);
  const auto& spline = getSpline(sector, row);
  // shrink to the grid
  if (val[0] < 0.f || val[0] > (float)spline.getGridX1().getUmax() || //
      val[1] < 0.f || val[1] > (float)spline.getGridX2().getUmax()) {
    return false;
  }
  return true;
}

GPUdi() std::array<float, 2> TPCFastSpaceChargeCorrection::convGridToLocal(int32_t sector, int32_t row, float gridU, float gridV) const
{
  /// convert internal grid coordinates u,v to local y, z
  return getSectorRowInfo(sector, row).gridMeasured.convGridToLocal(gridU, gridV);
}

GPUdi() std::array<float, 3> TPCFastSpaceChargeCorrection::convRealLocalToGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// convert real y, z to the internal grid coordinates + scale
  const SplineType& spline = getSpline(sector, row);
  auto val = getSectorRowInfo(sector, row).gridReal.convLocalToGridUntruncated(y, z);
  // shrink to the grid
  val[0] = GPUCommonMath::Clamp(val[0], 0.f, (float)spline.getGridX1().getUmax());
  val[1] = GPUCommonMath::Clamp(val[1], 0.f, (float)spline.getGridX2().getUmax());
  return val;
}

GPUdi() std::array<float, 2> TPCFastSpaceChargeCorrection::convGridToRealLocal(int32_t sector, int32_t row, float gridU, float gridV) const
{
  /// convert internal grid coordinates u,v to the real y, z
  return getSectorRowInfo(sector, row).gridReal.convGridToLocal(gridU, gridV);
}

GPUdi() std::array<float, 3> TPCFastSpaceChargeCorrection::getCorrectionLocal(int32_t sector, int32_t row, float y, float z) const
{
  const auto& info = getSectorRowInfo(sector, row);
  const SplineType& spline = getSpline(sector, row);
  const float* splineData = getCorrectionData(sector, row);

  auto val = convLocalToGrid(sector, row, y, z);

  float dxyz[3];
  spline.interpolateAtU(splineData, val[0], val[1], dxyz);

  float dx = val[2] * GPUCommonMath::Clamp(dxyz[0], info.minCorr[0], info.maxCorr[0]);
  float dy = val[2] * GPUCommonMath::Clamp(dxyz[1], info.minCorr[1], info.maxCorr[1]);
  float dz = val[2] * GPUCommonMath::Clamp(dxyz[2], info.minCorr[2], info.maxCorr[2]);
  return {dx, dy, dz};
}

GPUdi() float TPCFastSpaceChargeCorrection::getCorrectionXatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const
{
  const auto& info = getSectorRowInfo(sector, row);
  auto val = convRealLocalToGrid(sector, row, realY, realZ);
  float dx = 0;
  getSplineInvX(sector, row).interpolateAtU(getCorrectionDataInvX(sector, row), val[0], val[1], &dx);
  dx = val[2] * GPUCommonMath::Clamp(dx, info.minCorr[0], info.maxCorr[0]);
  return dx;
}

GPUdi() std::array<float, 2> TPCFastSpaceChargeCorrection::getCorrectionYZatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const
{
  auto val = convRealLocalToGrid(sector, row, realY, realZ);
  const auto& info = getSectorRowInfo(sector, row);
  float dyz[2];
  getSplineInvYZ(sector, row).interpolateAtU(getCorrectionDataInvYZ(sector, row), val[0], val[1], dyz);
  dyz[0] = val[2] * GPUCommonMath::Clamp(dyz[0], info.minCorr[1], info.maxCorr[1]);
  dyz[1] = val[2] * GPUCommonMath::Clamp(dyz[1], info.minCorr[2], info.maxCorr[2]);
  return {dyz[0], dyz[1]};
}

} // namespace gpu
} // namespace o2

#endif
