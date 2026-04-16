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
/// The TPCFastTransformGeoPOD class contains TPC geometry needed for the TPCFastTransformPOD
///
struct TPCFastTransformGeoPOD {
  /// Gives number of TPC sectors
  inline static constexpr int32_t getNumberOfSectors() { return GPUTPCGeometry::NSECTORS; }

  /// Gives number of TPC sectors on the A side
  inline static constexpr int32_t getNumberOfSectorsA() { return GPUTPCGeometry::NSECTORS / 2; }

  /// Gives number of TPC rows
  inline static constexpr int32_t getNumberOfRows() { return GPUTPCGeometry::NROWS; }

  /// Gives sector info
  inline static constexpr float getSectorSin(uint32_t sector) { return GPUTPCGeometry::SectorSin(sector); }
  inline static constexpr float getSectorCos(uint32_t sector) { return GPUTPCGeometry::SectorCos(sector); }

  /// Gives TPC row info
  inline static constexpr float getRowInfoX(uint32_t row) { return GPUTPCGeometry::Row2X(row); }
  inline static constexpr int32_t getRowInfoMaxPad(uint32_t row) { return GPUTPCGeometry::NPads(row) - 1; }
  inline static constexpr float getRowInfoPadWidth(uint32_t row) { return GPUTPCGeometry::PadWidth(row); }

  /// Gives Z length of the TPC, one Z side
  inline static constexpr float getTPCzLength() { return GPUTPCGeometry::TPCLength(); }

  /// Gives Z range for the corresponding TPC side
  inline static constexpr float getZmin(uint32_t sector) { return sector < getNumberOfSectorsA() ? 0.f : -getTPCzLength(); }
  inline static constexpr float getZmax(uint32_t sector) { return sector < getNumberOfSectorsA() ? getTPCzLength() : 0.f; }
  inline static constexpr float getZreadout(uint32_t sector) { return sector < getNumberOfSectorsA() ? getTPCzLength() : -getTPCzLength(); }

  /// _______________  Conversion of coordinate systems __________

  /// convert Local -> Global c.s.
  inline static constexpr void convLocalToGlobal(uint32_t sector, float lx, float ly, float lz, float& gx, float& gy, float& gz)
  {
    const float sinAlpha = getSectorSin(sector);
    const float cosAlpha = getSectorCos(sector);
    gx = lx * cosAlpha - ly * sinAlpha;
    gy = lx * sinAlpha + ly * cosAlpha;
    gz = lz;
  }

  /// convert Global->Local c.s.
  inline static constexpr void convGlobalToLocal(uint32_t sector, float gx, float gy, float gz, float& lx, float& ly, float& lz)
  {
    const float sinAlpha = getSectorSin(sector);
    const float cosAlpha = getSectorCos(sector);
    lx = gx * cosAlpha + gy * sinAlpha;
    ly = -gx * sinAlpha + gy * cosAlpha;
    lz = gz;
  }

  /// convert Pad, DriftLength -> Local c.s.
  inline static constexpr void convPadDriftLengthToLocal(uint32_t sector, uint32_t row, float pad, float driftLength, float& y, float& z)
  {
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

  /// convert DriftLength -> Local c.s.
  inline static constexpr float convDriftLengthToZ(uint32_t sector, float driftLength)
  {
    return (sector < getNumberOfSectorsA()) ? (getTPCzLength() - driftLength) : (driftLength - getTPCzLength());
  }

  /// convert Z to DriftLength
  inline static constexpr float convZtoDriftLength(uint32_t sector, float z)
  {
    return (sector < getNumberOfSectorsA()) ? (getTPCzLength() - z) : (z + getTPCzLength());
  }

  /// convert Local c.s. -> Pad, DriftLength
  inline static constexpr void convLocalToPadDriftLength(uint32_t sector, uint32_t row, float y, float z, float& pad, float& l)
  {
    /// convert Local c.s. -> Pad, DriftLength
    float u = 0;
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
};

} // namespace o2::gpu

#endif
