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

/// \file  TPCFastTransformGeoPOD.h
/// \brief Version using constexpr GPUTPCGeometry to be used for TPCFastTransformationPOD
///
/// \author David Rohr <drohr@cern.ch>

#ifndef ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMGEOPOD_H
#define ALICEO2_GPUCOMMON_TPCFASTTRANSFORMATION_TPCFASTTRANSFORMGEOPOD_H

#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "GPUTPCGeometry.h"

namespace o2::gpu
{

///
/// The TPCFastTransformGeoPOD class contains TPC geometry needed for the TPCFastTransform
///
struct TPCFastTransformGeoPOD {
  /// The struct contains necessary info for TPC sector
  struct SectorInfo {
    float sinAlpha{0.f}; ///< sin of the angle between the local x and the global x
    float cosAlpha{0.f}; ///< cos of the angle between the local x and the global x
  };

  /// The struct contains necessary info about TPC padrow
  struct RowInfo {
    float x{0.f};        ///< nominal X coordinate of the padrow [cm]
    int32_t maxPad{0};   ///< maximal pad number = n pads - 1
    float padWidth{0.f}; ///< width of pads [cm]
    float yMin{0.f};     ///< min. y coordinate

    /// get Y min
    GPUdi() float getYmin() const { return yMin; }

    /// get Y max
    GPUdi() float getYmax() const { return -yMin; }

    /// get width in Y
    GPUdi() float getYwidth() const { return -2.f * yMin; }
  };

  /// Gives number of TPC sectors
  inline static constexpr int32_t getNumberOfSectors() { return GPUTPCGeometry::NSECTORS; }

  /// Gives number of TPC sectors on the A side
  inline static constexpr int32_t getNumberOfSectorsA() { return GPUTPCGeometry::NSECTORS / 2; }

  /// Gives number of TPC rows
  GPUdi() int32_t getNumberOfRows() const { return GPUTPCGeometry::NROWS; }

  /// Gives sector info
  GPUd() const SectorInfo& getSectorInfo(uint32_t sector) const;

  /// Gives TPC row info
  GPUd() float getRowInfoX(uint32_t row) const { return GPUTPCGeometry::Row2X(row); }
  GPUd() int32_t getRowInfoMaxPad(uint32_t row) const { return GPUTPCGeometry::NPads(row) - 1; }
  GPUd() float getRowInfoPadWidth(uint32_t row) const { return GPUTPCGeometry::PadWidth(row); }

  /// Gives Z length of the TPC, one Z side
  GPUdi() float getTPCzLength() const { return GPUTPCGeometry::TPCLength(); }

  /// Gives Z range for the corresponding TPC side
  GPUd() float getZmin(uint32_t sector) const;
  GPUd() float getZmax(uint32_t sector) const;
  GPUd() float getZreadout(uint32_t sector) const;

  /// _______________  Conversion of coordinate systems __________

  /// convert Local -> Global c.s.
  GPUd() void convLocalToGlobal(uint32_t sector, float lx, float ly, float lz, float& gx, float& gy, float& gz) const;

  /// convert Global->Local c.s.
  GPUd() void convGlobalToLocal(uint32_t sector, float gx, float gy, float gz, float& lx, float& ly, float& lz) const;

  /// convert Pad, DriftLength -> Local c.s.
  GPUd() void convPadDriftLengthToLocal(uint32_t sector, uint32_t row, float pad, float driftLength, float& y, float& z) const;

  /// convert DriftLength -> Local c.s.
  GPUd() float convDriftLengthToZ1(uint32_t sector, float driftLength) const;

  /// convert Z to DriftLength
  GPUd() float convZtoDriftLength1(uint32_t sector, float z) const;

  /// convert Local c.s. -> Pad, DriftLength
  GPUd() void convLocalToPadDriftLength(uint32_t sector, uint32_t row, float y, float z, float& pad, float& l) const;

 private:
  /// _______________  Data members  _______________________________________________

  uint32_t mConstructionMask = 0;

  /// _______________  Geometry  _______________________________________________

  int32_t mNumberOfRows = 0; ///< Number of TPC rows. It is different for the Run2 and the Run3 setups
  float mTPCzLength = 0.f;   ///< Z length of one TPC side (A or C)

  SectorInfo mSectorInfos[GPUTPCGeometry::NSECTORS + 1]; ///< array of sector information [fixed size]
  RowInfo mRowInfos[160 + 1];                            ///< array of row information [fixed size]
};

// =======================================================================
//              Inline implementations of some methods
// =======================================================================

GPUdi() const TPCFastTransformGeoPOD::SectorInfo& TPCFastTransformGeoPOD::getSectorInfo(uint32_t sector) const
{
  return mSectorInfos[sector];
}

GPUdi() void TPCFastTransformGeoPOD::convLocalToGlobal(uint32_t sector, float lx, float ly, float lz, float& gx, float& gy, float& gz) const
{
  /// convert Local -> Global c.s.
  const SectorInfo& sectorInfo = getSectorInfo(sector);
  gx = lx * sectorInfo.cosAlpha - ly * sectorInfo.sinAlpha;
  gy = lx * sectorInfo.sinAlpha + ly * sectorInfo.cosAlpha;
  gz = lz;
}

GPUdi() void TPCFastTransformGeoPOD::convGlobalToLocal(uint32_t sector, float gx, float gy, float gz, float& lx, float& ly, float& lz) const
{
  /// convert Global -> Local c.s.
  const SectorInfo& sectorInfo = getSectorInfo(sector);
  lx = gx * sectorInfo.cosAlpha + gy * sectorInfo.sinAlpha;
  ly = -gx * sectorInfo.sinAlpha + gy * sectorInfo.cosAlpha;
  lz = gz;
}

GPUdi() void TPCFastTransformGeoPOD::convPadDriftLengthToLocal(uint32_t sector, uint32_t row, float pad, float driftLength, float& y, float& z) const
{
  /// convert Pad, DriftLength -> Local c.s.
  const float maxPad = getRowInfoMaxPad(row);
  const float padWidth = getRowInfoPadWidth(row);
  const float u = (pad - 0.5f * maxPad) * padWidth;
  if (sector < getNumberOfSectorsA()) { // TPC side A
    y = u;
    z = getTPCzLength() - driftLength;
  } else {                             // TPC side C
    y = -u;                            // pads are mirrorred on C-side
    z = driftLength - getTPCzLength(); // drift direction is mirrored on C-side
  }
}

GPUdi() float TPCFastTransformGeoPOD::convDriftLengthToZ1(uint32_t sector, float driftLength) const
{
  /// convert DriftLength -> Local c.s.
  return (sector < getNumberOfSectorsA()) ? (getTPCzLength() - driftLength) : (driftLength - getTPCzLength());
}

GPUdi() float TPCFastTransformGeoPOD::convZtoDriftLength1(uint32_t sector, float z) const
{
  /// convert Z to DriftLength
  return (sector < getNumberOfSectorsA()) ? (getTPCzLength() - z) : (z + getTPCzLength());
}

GPUdi() float TPCFastTransformGeoPOD::getZmin(uint32_t sector) const
{
  /// z min for the sector
  if (sector < getNumberOfSectorsA()) { // TPC side A
    return 0.f;
  } else { // TPC side C
    return -getTPCzLength();
  }
}

GPUdi() float TPCFastTransformGeoPOD::getZmax(uint32_t sector) const
{
  /// z max for the sector
  if (sector < getNumberOfSectorsA()) { // TPC side A
    return getTPCzLength();
  } else { // TPC side C
    return 0.f;
  }
}

GPUdi() float TPCFastTransformGeoPOD::getZreadout(uint32_t sector) const
{
  /// z readout for the sector
  if (sector < getNumberOfSectorsA()) { // TPC side A
    return getTPCzLength();
  } else { // TPC side C
    return -getTPCzLength();
  }
}

GPUdi() void TPCFastTransformGeoPOD::convLocalToPadDriftLength(uint32_t sector, uint32_t row, float y, float z, float& pad, float& l) const
{
  /// convert Local c.s. -> Pad, DriftLength
  float u;
  if (sector < getNumberOfSectorsA()) { // TPC side A
    u = y;
    l = getTPCzLength() - z;
  } else {                   // TPC side C
    u = -y;                  // pads are mirrorred on C-side
    l = z + getTPCzLength(); // drift direction is mirrored on C-side
  }
  const float maxPad = getRowInfoMaxPad(row);
  const float padWidth = getRowInfoPadWidth(row);
  pad = u / padWidth + 0.5f * maxPad;
}

} // namespace o2::gpu

#endif
