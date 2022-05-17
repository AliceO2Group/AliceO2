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

/// \file RecoGeomHelper.h
/// \brief Declarations of the helper class for clusters / roadwidth matching
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_ITS_RECOGEOMHELPER_H
#define ALICEO2_ITS_RECOGEOMHELPER_H

#include <Rtypes.h>
#include <vector>
#include <array>
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "MathUtils/Primitive2D.h"
#include "CommonConstants/MathConstants.h"
#include "MathUtils/Primitive2D.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"
#include "ITSMFTBase/SegmentationAlpide.h"

namespace o2
{
namespace its
{
struct RecoGeomHelper {
  //
  using BracketF = o2::math_utils::Bracketf_t;
  using Vec2D = o2::math_utils::IntervalXYf_t;

  enum Relation : int { Below = -1,
                        Inside = 0,
                        Above = 1 };

  struct RecoChip {
    ///< boundaries + frame data for single chip
    uint16_t id = 0xffff;                                    // global chip id
    float alp = 0.f, snAlp = 0.f, csAlp = 0.f;               // cos and sin of sensor alpha
    float xRef = 0.f;                                        // reference X
    BracketF yRange = {1.e9, -1.e9}, zRange = {1.e9, -1.e9}; // bounding box in tracking frame
    Vec2D xyEdges;

    void updateLimits(const o2::math_utils::Point3D<float>& pnt);
    void print() const;
    ClassDefNV(RecoChip, 0);
  };

  struct RecoLadder {
    enum Overlap : int8_t { Undefined,
                            Above,
                            NoOverlap,
                            Below };
    ///< group of chips at same (modulo alignment) phi, r (stave in IB, 1/4 stave in OB)
    short id = 0;                        // assigned ladder ID within the layer
    Overlap overlapWithNext = Undefined; // special flag saying if for double-hit layers this one is above, excluded or below the next one
    BracketF phiRange = {o2::constants::math::TwoPI, 0.};
    BracketF zRange = {1e9, -1e9}; // Z ranges in lab frame
    Vec2D xyEdges;                 // envelop for chip edges
    float phiMean = 0., dphiH = 0.;
    std::vector<RecoChip> chips;

    Relation isPhiOutside(float phi, float toler = 0) const;
    void updateLimits(const o2::math_utils::Point3D<float>& pnt);
    void init();
    void print() const;
    ClassDefNV(RecoLadder, 0);
  };

  struct RecoLayer {
    // navigation over ladders of single layer
    int id = 0;       // layer ID
    int nLadders = 0; // number of ladders
    int lastChipInLadder = 0;
    float phi2bin = 0;
    float z2chipID = 0;            // conversion factor for Z (relative to zmin) to rough chip ID
    float rInv = 0.;               // inverse mean radius
    BracketF rRange = {1e9, 0.};   // min and max radii
    BracketF zRange = {1e9, -1e9}; // min and max Z
    std::vector<RecoLadder> ladders;
    std::vector<uint16_t> phi2ladder; // mapping from phi to ladderID

    const RecoLadder& getLadder(int id) const { return ladders[id % nLadders]; }
    void init();
    void updateLimits(const o2::math_utils::Point3D<float>& pnt);
    void print() const;
    int getLadderID(float phi) const;
    int getChipID(float z) const;
    ClassDefNV(RecoLayer, 0);
  };
  //---------------------------<< aux classes

  std::array<RecoLayer, o2::itsmft::ChipMappingITS::NLayers> layers;
  static constexpr int getNLayers() { return o2::itsmft::ChipMappingITS::NLayers; }
  static constexpr int getNChips() { return o2::itsmft::ChipMappingITS::getNChips(); }
  static constexpr float ladderWidth() { return o2::itsmft::SegmentationAlpide::SensorSizeRows; }
  static constexpr float ladderWidthInv() { return 1. / ladderWidth(); }

  void init();
  void print() const;

  ClassDefNV(RecoGeomHelper, 0);
};

//_____________________________________________________________________
inline RecoGeomHelper::Relation RecoGeomHelper::RecoLadder::isPhiOutside(float phi, float toler) const
{
  // check if phi+-toler is out of the limits of the ladder phi.
  // return -1 or +1 if phi is above or below the region. If inside, return 0
  float dif = phi - phiMean, difa = fabs(dif);
  if (difa < dphiH + toler) {
    return RecoGeomHelper::Inside;
  }
  if (difa > o2::constants::math::PI) { // wraps?
    difa = o2::constants::math::TwoPI - difa;
    if (difa < dphiH + toler) {
      return RecoGeomHelper::Inside;
    }
    return dif < 0 ? RecoGeomHelper::Above : RecoGeomHelper::Below;
  }
  return dif < 0 ? RecoGeomHelper::Below : RecoGeomHelper::Above;
}

//_____________________________________________________________________
inline int RecoGeomHelper::RecoLayer::getChipID(float z) const
{
  // Get chip ID within the ladder corresponding to this phi
  // Note: this is an approximate method, one should check also the neighbouring ladders +/-1
  int ic = (z - zRange.getMin()) * z2chipID;
  return ic < 0 ? 0 : (ic < lastChipInLadder ? ic : lastChipInLadder);
}

//_____________________________________________________________________
inline int RecoGeomHelper::RecoLayer::getLadderID(float phi) const
{
  // Get ladder ID corresponding to phi.
  // Note: this is an approximate method, precise within 1/3 of average ladder width,
  // one should check also the neighbouring ladders +/-1
  o2::math_utils::bringTo02Pi(phi);
  return phi2ladder[int(phi * phi2bin)];
}

} // namespace its
} // namespace o2

#endif /* ALICEO2_ITS_RECOGEOMHELPER_H */
