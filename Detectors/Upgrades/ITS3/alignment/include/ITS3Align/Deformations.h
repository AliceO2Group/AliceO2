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

#ifndef ITS3_DEFORMATIONS_H_
#define ITS3_DEFORMATIONS_H_

#include "ITS3Align/MisalignmentParameters.h"
#include "MathUtils/LegendrePols.h"

#include <filesystem>

namespace o2::its3::align
{

class Deformations
{
 public:
  // init deformations from the parameter file
  void init(const std::filesystem::path&);

  double getDeformationX(unsigned int id, double u, double v) const { return getDeformation<0>(id, u, v); }
  double getDeformationY(unsigned int id, double u, double v) const { return getDeformation<1>(id, u, v); }
  double getDeformationZ(unsigned int id, double u, double v) const { return getDeformation<2>(id, u, v); }
  double getDeformation(unsigned int id, unsigned int axis, double u, double v) const
  {
    if (axis == 0) {
      return mLegX[id](u, v);
    } else if (axis == 1) {
      return mLegY[id](u, v);
    } else {
      return mLegZ[id](u, v);
    }
  }
  std::array<double, 3> getDeformation(unsigned int id, double u, double v) const
  {
    return {getDeformation<0>(id, u, v),
            getDeformation<1>(id, u, v),
            getDeformation<2>(id, u, v)};
  }
  std::array<unsigned int, 3> getOrders(unsigned int id) const
  {
    return {mLegX[id].NOrder(), mLegY[id].NOrder(), mLegZ[id].NOrder()};
  }
  const o2::math_utils::Legendre2DPolynominal& getLegendre(unsigned int id, unsigned int axis) const
  {
    if (axis == 0) {
      return mLegX[id];
    } else if (axis == 1) {
      return mLegY[id];
    } else {
      return mLegZ[id];
    }
  }

 private:
  template <int axis>
  double getDeformation(unsigned int id, double u, double v) const
  {
    if constexpr (axis == 0) {
      return mLegX[id](u, v);
    } else if constexpr (axis == 1) {
      return mLegY[id](u, v);
    } else {
      return mLegZ[id](u, v);
    }
  }

  // 3 Legendre polynominals to model deformations in x,y,z; parameterized by normalized phi (u) and z (v) coordinates
  std::array<o2::math_utils::Legendre2DPolynominal, 6> mLegX;
  std::array<o2::math_utils::Legendre2DPolynominal, 6> mLegY;
  std::array<o2::math_utils::Legendre2DPolynominal, 6> mLegZ;
};

} // namespace o2::its3::align

#endif
