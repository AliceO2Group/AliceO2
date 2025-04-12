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

/// \file  TPCFastTransformGeo.h
/// \brief Definition of TPCFastTransformGeo class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMGEO_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMGEO_H

#include "GPUCommonDef.h"
#ifndef GPUCA_GPUCODE_DEVICE
#include <memory>
#include "GPUCommonRtypes.h"
#endif

namespace o2
{
namespace gpu
{

///
/// The TPCFastTransformGeo class contains TPC geometry needed for the TPCFastTransform
///
class TPCFastTransformGeo
{
 public:
  /// The struct contains necessary info for TPC sector
  struct SectorInfo {
    float sinAlpha{0.f}; ///< sin of the angle between the local x and the global x
    float cosAlpha{0.f}; ///< cos of the angle between the local x and the global x
    ClassDefNV(SectorInfo, 1);
  };

  /// The struct contains necessary info about TPC padrow
  struct RowInfo {
    float x{0.f};        ///< nominal X coordinate of the padrow [cm]
    int32_t maxPad{0};   ///< maximal pad number = n pads - 1
    float padWidth{0.f}; ///< width of pads [cm]
    float yMin{0.f};     ///< min. y coordinate

    /// get Y min
    GPUd() float getYmin() const { return yMin; }

    /// get Y max
    GPUd() float getYmax() const { return -yMin; }

    /// get Y range
    GPUd() std::tuple<float, float> getYrange() const { return {getYmin(), getYmax()}; }

    /// get width in Y
    GPUd() float getYwidth() const { return -2.f * yMin; }

    ClassDefNV(RowInfo, 1);
  };

  /// _____________  Constructors / destructors __________________________

  /// Default constructor: creates an empty uninitialized object
  TPCFastTransformGeo();

  /// Copy constructor: disabled to avoid ambiguity. Use cloneFromObject() instead
  TPCFastTransformGeo(const TPCFastTransformGeo&) = default;

  /// Assignment operator: disabled to avoid ambiguity. Use cloneFromObject() instead
  TPCFastTransformGeo& operator=(const TPCFastTransformGeo&) = default;

  /// Destructor
  ~TPCFastTransformGeo() = default;

  /// _____________  FlatObject functionality, see FlatObject class for description  ____________

  /// Gives minimal alignment in bytes required for an object of the class
  static constexpr size_t getClassAlignmentBytes() { return 8; }

  /// _______________  Construction interface  ________________________

  /// Starts the initialization psectoredure, reserves temporary memory
  void startConstruction(int32_t numberOfRows);

  /// Initializes a TPC row
  void setTPCrow(int32_t iRow, float x, int32_t nPads, float padWidth);

  /// Sets TPC geometry
  ///
  /// It must be called once during initialization
  void setTPCzLength(float tpcZlength);

  /// Finishes initialization: puts everything to the flat buffer, releases temporary memory
  void finishConstruction();

  /// Is the object constructed
  bool isConstructed() const { return (mConstructionMask == (uint32_t)ConstructionState::Constructed); }

  /// _______________  Getters _________________________________

  /// Gives number of TPC sectors
  GPUd() static constexpr int32_t getNumberOfSectors() { return NumberOfSectors; }

  /// Gives number of TPC sectors on the A side
  GPUd() static constexpr int32_t getNumberOfSectorsA() { return NumberOfSectorsA; }

  /// Gives number of TPC rows
  GPUd() int32_t getNumberOfRows() const { return mNumberOfRows; }

  /// Gives number of TPC rows
  GPUd() static constexpr int getMaxNumberOfRows() { return MaxNumberOfRows; }

  /// Gives sector info
  GPUd() const SectorInfo& getSectorInfo(int32_t sector) const;

  /// Gives TPC row info
  GPUd() const RowInfo& getRowInfo(int32_t row) const;

  /// Gives Z length of the TPC, one Z side
  GPUd() float getTPCzLength() const { return mTPCzLength; }

  /// Gives Z range for the corresponding TPC side
  GPUd() std::tuple<float, float> getZrange(int32_t sector) const;

  /// _______________  Conversion of coordinate systems __________

  /// convert Local -> Global c.s.
  GPUd() void convLocalToGlobal(int32_t sector, float lx, float ly, float lz, float& gx, float& gy, float& gz) const;

  /// convert Global->Local c.s.
  GPUd() void convGlobalToLocal(int32_t sector, float gx, float gy, float gz, float& lx, float& ly, float& lz) const;

  /// convert Pad, DriftLength -> Local c.s.
  GPUd() std::tuple<float, float> convPadDriftLengthToLocal(int32_t sector, int32_t row, float pad, float driftLength) const;

  /// convert DriftLength -> Local c.s.
  GPUd() float convDriftLengthToZ(int32_t sector, float driftLength) const;

  /// convert Z to DriftLength
  GPUd() float convZtoDriftLength(int32_t sector, float z) const;

  /// convert Local c.s. -> Pad, DriftLength
  GPUd() std::tuple<float, float> convLocalToPadDriftLength(int32_t sector, int32_t row, float y, float z) const;

  /// Print method
  void print() const;

  /// Method for testing consistency
  int32_t test(int32_t sector, int32_t row, float ly, float lz) const;

  /// Method for testing consistency
  int32_t test() const;

 private:
  /// _______________  Data members  _______________________________________________

  static constexpr int32_t NumberOfSectors = 36;                   ///< Number of TPC sectors ( sector = inner + outer sector )
  static constexpr int32_t NumberOfSectorsA = NumberOfSectors / 2; ///< Number of TPC sectors side A
  static constexpr int32_t MaxNumberOfRows = 160;                  ///< Max Number of TPC rows in a sector

  /// _______________  Construction control  _______________________________________________

  /// Enumeration of construction states
  enum ConstructionState : uint32_t {
    NotConstructed = 0x0, ///< the object is not constructed
    Constructed = 0x1,    ///< the object is constructed, temporary memory is released
    InProgress = 0x2,     ///< construction started: temporary  memory is reserved
    GeometryIsSet = 0x4,  ///< the TPC geometry is set
  };

  uint32_t mConstructionMask = ConstructionState::NotConstructed; ///< mask for constructed object members, first two bytes are used by this class

  /// _______________  Geometry  _______________________________________________

  int32_t mNumberOfRows = 0; ///< Number of TPC rows. It is different for the Run2 and the Run3 setups
  float mTPCzLength = 0.f;   ///< Z length of one TPC side (A or C)

  SectorInfo mSectorInfos[NumberOfSectors + 1]; ///< array of sector information [fixed size]
  RowInfo mRowInfos[MaxNumberOfRows + 1]; ///< array of row information [fixed size]

  ClassDefNV(TPCFastTransformGeo, 3);
};

// =======================================================================
//              Inline implementations of some methods
// =======================================================================

GPUdi() const TPCFastTransformGeo::SectorInfo& TPCFastTransformGeo::getSectorInfo(int32_t sector) const
{
  /// Gives sector info
  if (sector < 0 || sector >= NumberOfSectors) { // return zero object
    sector = NumberOfSectors;
  }
  return mSectorInfos[sector];
}

GPUdi() const TPCFastTransformGeo::RowInfo& TPCFastTransformGeo::getRowInfo(int32_t row) const
{
  /// Gives TPC row info
  if (row < 0 || row >= mNumberOfRows) { // return zero object
    row = MaxNumberOfRows;
  }
  return mRowInfos[row];
}

GPUdi() void TPCFastTransformGeo::convLocalToGlobal(int32_t sector, float lx, float ly, float lz, float& gx, float& gy, float& gz) const
{
  /// convert Local -> Global c.s.
  const SectorInfo& sectorInfo = getSectorInfo(sector);
  gx = lx * sectorInfo.cosAlpha - ly * sectorInfo.sinAlpha;
  gy = lx * sectorInfo.sinAlpha + ly * sectorInfo.cosAlpha;
  gz = lz;
}

GPUdi() void TPCFastTransformGeo::convGlobalToLocal(int32_t sector, float gx, float gy, float gz, float& lx, float& ly, float& lz) const
{
  /// convert Global -> Local c.s.
  const SectorInfo& sectorInfo = getSectorInfo(sector);
  lx = gx * sectorInfo.cosAlpha + gy * sectorInfo.sinAlpha;
  ly = -gx * sectorInfo.sinAlpha + gy * sectorInfo.cosAlpha;
  lz = gz;
}

GPUdi() std::tuple<float, float> TPCFastTransformGeo::convPadDriftLengthToLocal(int32_t sector, int32_t row, float pad, float driftLength) const
{
  /// convert Pad, DriftLength -> Local c.s.
  const RowInfo& rowInfo = getRowInfo(row);
  float u = (pad - 0.5f * rowInfo.maxPad) * rowInfo.padWidth;
  float y, z;
  if (sector < NumberOfSectorsA) { // TPC side A
    y = u;
    z = mTPCzLength - driftLength;
  } else {                         // TPC side C
    y = -u;                        // pads are mirrorred on C-side
    z = driftLength - mTPCzLength; // drift direction is mirrored on C-side
  }
  return {y, z};
}

GPUdi() float TPCFastTransformGeo::convDriftLengthToZ(int32_t sector, float driftLength) const
{
  /// convert DriftLength -> Local c.s.
  return (sector < NumberOfSectorsA) ? (mTPCzLength - driftLength) : (driftLength - mTPCzLength);
}

GPUdi() float TPCFastTransformGeo::convZtoDriftLength(int32_t sector, float z) const
{
  /// convert Z to DriftLength
  return (sector < NumberOfSectorsA) ? (mTPCzLength - z) : (z + mTPCzLength);
}

GPUdi() std::tuple<float, float> TPCFastTransformGeo::getZrange(int32_t sector) const
{
  /// z range for the sector
  if (sector < NumberOfSectorsA) { // TPC side A
    return {0.f, mTPCzLength};
  } else { // TPC side C
    return {-mTPCzLength, 0.f};
  }
}

GPUdi() std::tuple<float, float> TPCFastTransformGeo::convLocalToPadDriftLength(int32_t sector, int32_t row, float y, float z) const
{
  /// convert Local c.s. -> Pad, DriftLength
  float u, l;
  if (sector < NumberOfSectorsA) { // TPC side A
    u = y;
    l = mTPCzLength - z;
  } else {               // TPC side C
    u = -y;              // pads are mirrorred on C-side
    l = z + mTPCzLength; // drift direction is mirrored on C-side
  }
  const TPCFastTransformGeo::RowInfo& rowInfo = getRowInfo(row);
  float pad = u / rowInfo.padWidth + 0.5f * rowInfo.maxPad;
  return {pad, l};
}

} // namespace gpu
} // namespace o2

#endif
