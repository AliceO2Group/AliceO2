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
  ///
  /// \brief The struct contains necessary info for TPC padrow
  ///
  struct RowInfo {
    int32_t splineScenarioID{0};  ///< scenario index (which of Spline2D splines to use)
    size_t dataOffsetBytes[3]{0}; ///< offset for the spline data withing the TPC sector
    ClassDefNV(RowInfo, 1);
  };

  struct SectorRowInfo {
    float gridU0{0.f};                     //< U coordinate of the U-grid start
    float scaleUtoGrid{0.f};               //< scale U to U-grid coordinate
    float gridV0{0.f};                     ///< V coordinate of the V-grid start
    float scaleVtoGrid{0.f};               //< scale V to V-grid coordinate
    float gridCorrU0{0.f};                 ///< U coordinate of the U-grid start for corrected U
    float scaleCorrUtoGrid{0.f};           ///< scale corrected U to U-grid coordinate
    float gridCorrV0{0.f};                 ///< V coordinate of the V-grid start for corrected V
    float scaleCorrVtoGrid{0.f};           ///< scale corrected V to V-grid coordinate
    float minCorr[3]{-10.f, -10.f, -10.f}; ///< min correction for dX, dU, dV
    float maxCorr[3]{10.f, 10.f, 10.f};    ///< max correction for dX, dU, dV

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

    ClassDefNV(SectorRowInfo, 2);
  };

  struct SectorInfo {
    float vMax{0.f}; ///< Max value of V coordinate
    ClassDefNV(SectorInfo, 1);
  };

  typedef Spline2D<float, 3> SplineType;

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

  void setActualBufferAddress(char* actualFlatBufferPtr);
  void setFutureBufferAddress(char* futureFlatBufferPtr);

  /// _______________  Construction interface  ________________________

  /// Starts the construction procedure, reserves temporary memory
  void startConstruction(const TPCFastTransformGeo& geo, int32_t numberOfSplineScenarios);

  /// Initializes a TPC row
  void setRowScenarioID(int32_t iRow, int32_t iScenario);

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

  /// Set safety marging for the interpolation around the TPC row.
  /// Outside of this area the interpolation returns the boundary values.
  GPUd() void setInterpolationSafetyMargin(float val) { fInterpolationSafetyMargin = val; }

  /// Gives const pointer to a spline
  GPUd() const SplineType& getSpline(int32_t sector, int32_t row) const;

  /// Gives pointer to a spline
  GPUd() SplineType& getSpline(int32_t sector, int32_t row);

  /// Gives pointer to spline data
  GPUd() float* getSplineData(int32_t sector, int32_t row, int32_t iSpline = 0);

  /// Gives pointer to spline data
  GPUd() const float* getSplineData(int32_t sector, int32_t row, int32_t iSpline = 0) const;

  /// _______________ The main method: cluster correction  _______________________
  ///
  // GPUd() int32_t getCorrectionInternal(int32_t sector, int32_t row, float u, float v, float& dx, float& du, float& dv) const;

  GPUdi() std::tuple<float, float, float> getCorrectionLocal(int32_t sector, int32_t row, float y, float z) const;

  /// inverse correction: Corrected U and V -> coorrected X
  GPUd() float getCorrectionXatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const;

  /// inverse correction: Corrected U and V -> uncorrected U and V
  GPUd() std::tuple<float, float> getCorrectionYZatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const;

  /// _______________  Utilities  _______________________________________________

  /// convert local y, z to internal grid coordinates u,v
  /// return values: u, v, scaling factor
  GPUd() std::tuple<float, float, float> convLocalToGrid(int32_t sector, int32_t row, float y, float z) const;

  /// convert internal grid coordinates u,v to local y, z
  /// return values: y, z, scaling factor
  GPUd() std::tuple<float, float> convGridToLocal(int32_t sector, int32_t row, float u, float v) const;

  /// convert corrected u,v to internal grid coordinates
  GPUd() std::tuple<float, float, float> convCorrectedLocalToGrid(int32_t sector, int32_t row, float y, float z) const;

  GPUd() bool isLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const;

  /// TPC geometry information
  GPUd() const TPCFastTransformGeo& getGeometry() const
  {
    return mGeo;
  }

  /// Gives the time stamp of the current calibaration parameters
  int64_t getTimeStamp() const { return mTimeStamp; }

  /// Gives the interpolation safety marging  around the TPC row.
  GPUd() float getInterpolationSafetyMargin() const { return fInterpolationSafetyMargin; }

  /// Gives TPC row info
  GPUd() const RowInfo& getRowInfo(int32_t row) const { return mRowInfos[row]; }

  /// Gives TPC sector info
  GPUd() const SectorInfo& getSectorInfo(int32_t sector) const
  {
    return mSectorInfo[sector];
  }

  /// Gives TPC sector info
  GPUd() SectorInfo& getSectorInfo(int32_t sector)
  {
    return mSectorInfo[sector];
  }

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

  SectorInfo mSectorInfo[TPCFastTransformGeo::getNumberOfSectors()]; ///< SectorInfo array

  SplineType* mScenarioPtr; //! (transient!!) pointer to spline scenarios

  /// _______________  Calibration data  _______________________________________________

  int64_t mTimeStamp; ///< time stamp of the current calibration

  char* mSplineData[3]; //! (transient!!) pointer to the spline data in the flat buffer

  size_t mSectorDataSizeBytes[3]; ///< size of the data for one sector in the flat buffer

  float fInterpolationSafetyMargin{0.1f}; // 10% area around the TPC row. Outside of this area the interpolation returns the boundary values.

  /// Class version. It is used to read older versions from disc.
  /// The default version 3 is the one before this field was introduced.
  /// The actual version must be set in startConstruction().
  int32_t mClassVersion{3};

  RowInfo mRowInfos[TPCFastTransformGeo::getMaxNumberOfRows()]; ///< RowInfo array

  SectorRowInfo mSectorRowInfos[TPCFastTransformGeo::getNumberOfSectors() * TPCFastTransformGeo::getMaxNumberOfRows()]; ///< SectorRowInfo array

  ClassDefNV(TPCFastSpaceChargeCorrection, 5);
};

/// ====================================================
///       Inline implementations of some methods
/// ====================================================

GPUdi() const TPCFastSpaceChargeCorrection::SplineType& TPCFastSpaceChargeCorrection::getSpline(int32_t sector, int32_t row) const
{
  /// Gives const pointer to spline
  const RowInfo& rowInfo = mRowInfos[row];
  return mScenarioPtr[rowInfo.splineScenarioID];
}

GPUdi() TPCFastSpaceChargeCorrection::SplineType& TPCFastSpaceChargeCorrection::getSpline(int32_t sector, int32_t row)
{
  /// Gives pointer to spline
  const RowInfo& rowInfo = mRowInfos[row];
  return mScenarioPtr[rowInfo.splineScenarioID];
}

GPUdi() float* TPCFastSpaceChargeCorrection::getSplineData(int32_t sector, int32_t row, int32_t iSpline)
{
  /// Gives pointer to spline data
  const RowInfo& rowInfo = mRowInfos[row];
  return reinterpret_cast<float*>(mSplineData[iSpline] + mSectorDataSizeBytes[iSpline] * sector + rowInfo.dataOffsetBytes[iSpline]);
}

GPUdi() const float* TPCFastSpaceChargeCorrection::getSplineData(int32_t sector, int32_t row, int32_t iSpline) const
{
  /// Gives pointer to spline data
  const RowInfo& rowInfo = mRowInfos[row];
  return reinterpret_cast<float*>(mSplineData[iSpline] + mSectorDataSizeBytes[iSpline] * sector + rowInfo.dataOffsetBytes[iSpline]);
}

GPUdi() std::tuple<float, float, float> TPCFastSpaceChargeCorrection::convLocalToGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// convert local y, z to internal grid coordinates u,v
  /// return values: u, v, scaling factor
  const auto& info = getSectorRowInfo(sector, row);
  const SplineType& spline = getSpline(sector, row);

  float u, v;
  mGeo.convLocalToUV1(sector, y, z, u, v);

  float scale = 1.f;
  if (v < 0.f) {
    scale = 0.f;
  } else if (v < info.gridV0) {
    scale = v / info.gridV0;
  }

  float gridU = (u - info.gridU0) * info.scaleUtoGrid;
  float gridV = (v - info.gridV0) * info.scaleVtoGrid;

  // shrink to the grid area
  gridU = GPUCommonMath::Clamp(gridU, 0.f, (float)spline.getGridX1().getUmax());
  gridV = GPUCommonMath::Clamp(gridV, 0.f, (float)spline.getGridX2().getUmax());

  return {gridU, gridV, scale};
}

GPUdi() bool TPCFastSpaceChargeCorrection::isLocalInsideGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// ccheck if local y, z are inside the grid

  const auto& info = getSectorRowInfo(sector, row);
  const SplineType& spline = getSpline(sector, row);

  float u, v;
  mGeo.convLocalToUV1(sector, y, z, u, v);

  float gridU = (u - info.gridU0) * info.scaleUtoGrid;
  float gridV = (v - info.gridV0) * info.scaleVtoGrid;

  // shrink to the grid area
  if (gridU < 0.f || gridU > (float)spline.getGridX1().getUmax())
    return false;
  if (gridV < 0.f || gridV > (float)spline.getGridX2().getUmax())
    return false;
  return true;
}

GPUdi() std::tuple<float, float> TPCFastSpaceChargeCorrection::convGridToLocal(int32_t sector, int32_t row, float gridU, float gridV) const
{
  /// convert internal grid coordinates u,v to local y, z
  const SectorRowInfo& info = getSectorRowInfo(sector, row);
  float u = info.gridU0 + gridU / info.scaleUtoGrid;
  float v = info.gridV0 + gridV / info.scaleVtoGrid;
  float y, z;
  mGeo.convUVtoLocal1(sector, u, v, y, z);
  return {y, z};
}

GPUdi() std::tuple<float, float, float> TPCFastSpaceChargeCorrection::convCorrectedLocalToGrid(int32_t sector, int32_t row, float y, float z) const
{
  /// convert corrected y, z to the internal grid coordinates
  const auto& info = getSectorRowInfo(sector, row);
  const Spline2D<float, 1>& spline = reinterpret_cast<const Spline2D<float, 1>&>(getSpline(sector, row));

  float u, v;
  mGeo.convLocalToUV1(sector, y, z, u, v);

  float scale = 1.f;
  if (v < 0.f) {
    scale = 0.f;
  } else if (v < info.gridCorrV0) {
    scale = v / info.gridCorrV0;
  }

  float gridU = (u - info.gridCorrU0) * info.scaleCorrUtoGrid;
  float gridV = (v - info.gridCorrV0) * info.scaleCorrVtoGrid;

  // shrink to the grid area
  gridU = GPUCommonMath::Clamp(gridU, 0.f, (float)spline.getGridX1().getUmax());
  gridV = GPUCommonMath::Clamp(gridV, 0.f, (float)spline.getGridX2().getUmax());

  return {gridU, gridV, scale};
}

GPUdi() std::tuple<float, float, float> TPCFastSpaceChargeCorrection::getCorrectionLocal(int32_t sector, int32_t row, float y, float z) const
{
  const auto& info = getSectorRowInfo(sector, row);
  const SplineType& spline = getSpline(sector, row);
  const float* splineData = getSplineData(sector, row);

  auto [gridU, gridV, scale] = convLocalToGrid(sector, row, y, z);

  float dxyz[3];
  spline.interpolateAtU(splineData, gridU, gridV, dxyz);

  float dx = scale * GPUCommonMath::Clamp(dxyz[0], info.minCorr[0], info.maxCorr[0]);
  float dy = scale * GPUCommonMath::Clamp(dxyz[1], info.minCorr[1], info.maxCorr[1]);
  float dz = scale * GPUCommonMath::Clamp(dxyz[2], info.minCorr[2], info.maxCorr[2]);
  return {dx, dy, dz};
}

GPUdi() float TPCFastSpaceChargeCorrection::getCorrectionXatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const
{
  const auto& info = getSectorRowInfo(sector, row);
  const Spline2D<float, 1>& spline = reinterpret_cast<const Spline2D<float, 1>&>(getSpline(sector, row));
  const float* splineData = getSplineData(sector, row, 1);

  auto [gridU, gridV, scale] = convCorrectedLocalToGrid(sector, row, realY, realZ);

  float dx = 0;
  spline.interpolateAtU(splineData, gridU, gridV, &dx);

  dx = scale * GPUCommonMath::Clamp(dx, info.minCorr[0], info.maxCorr[0]);
  return dx;
}

GPUdi() std::tuple<float, float> TPCFastSpaceChargeCorrection::getCorrectionYZatRealYZ(int32_t sector, int32_t row, float realY, float realZ) const
{

  auto [gridU, gridV, scale] = convCorrectedLocalToGrid(sector, row, realY, realZ);

  const auto& info = getSectorRowInfo(sector, row);
  const Spline2D<float, 2>& spline = reinterpret_cast<const Spline2D<float, 2>&>(getSpline(sector, row));
  const float* splineData = getSplineData(sector, row, 2);

  float dyz[2];
  spline.interpolateAtU(splineData, gridU, gridV, dyz);

  dyz[0] = scale * GPUCommonMath::Clamp(dyz[0], info.minCorr[1], info.maxCorr[1]);
  dyz[1] = scale * GPUCommonMath::Clamp(dyz[1], info.minCorr[2], info.maxCorr[2]);

  return {dyz[0], dyz[1]};
}

} // namespace gpu
} // namespace o2

#endif
