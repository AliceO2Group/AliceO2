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

/// \file MatCell.h
/// \brief Declarations for material properties of the cell (voxel)

#ifndef ALICEO2_MATCELL_H
#define ALICEO2_MATCELL_H

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"

namespace o2
{
namespace base
{

struct MatCell {
  // small struct to hold <X/X0> and <rho> of the voxel

  static constexpr int NParams = 2; // number of material parameters described
  float meanRho;                    ///< mean density, g/cm^3
  float meanX2X0;                   ///< fraction of radiaton lenght

  GPUd() MatCell() : meanRho(0.f), meanX2X0(0.f) {}
  GPUdDefault() MatCell(const MatCell& src) CON_DEFAULT;

  GPUd() void set(const MatCell& c)
  {
    meanRho = c.meanRho;
    meanX2X0 = c.meanX2X0;
  }

  GPUd() void scale(float scale)
  {
    meanRho *= scale;
    meanX2X0 *= scale;
  }

  ClassDefNV(MatCell, 1);
};

struct MatBudget : MatCell {

  // small struct to hold <X/X0>, <rho> and length traversed by track in the voxel
  static constexpr int NParams = 3; // number of material parameters described
  float length;                     ///< length in material

  GPUd() MatBudget() : length(0.f) {}
  GPUdDefault() MatBudget(const MatBudget& src) CON_DEFAULT;

  GPUd() void scale(float scale)
  {
    MatCell::scale(scale);
    length *= scale;
  }

  GPUd() float getXRho() const
  {
    return meanRho * length;
  }

  GPUd() float getXRho(int signCorr) const
  {
    return meanRho * (signCorr < 0 ? -length : length);
  }

  ClassDefNV(MatBudget, 1);
};

} // namespace base
} // namespace o2

#endif
