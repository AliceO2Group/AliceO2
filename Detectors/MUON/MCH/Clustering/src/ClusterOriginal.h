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

/// \file ClusterOriginal.h
/// \brief Definition of the cluster used by the original cluster finder algorithm
///
/// \author Philippe Pillot, Subatech

#ifndef O2_MCH_CLUSTERORIGINAL_H_
#define O2_MCH_CLUSTERORIGINAL_H_

#include "PadOriginal.h"
#include <utility>
#include <vector>

#include <TMath.h>

namespace o2
{
namespace mch
{

/// cluster for internal use
class ClusterOriginal
{
 public:
  ClusterOriginal() = default;
  ~ClusterOriginal() = default;

  ClusterOriginal(const ClusterOriginal& cl) = default;
  ClusterOriginal& operator=(const ClusterOriginal& cl) = default;
  ClusterOriginal(ClusterOriginal&&) = delete;
  ClusterOriginal& operator=(ClusterOriginal&&) = delete;

  void clear();

  void addPad(double x, double y, double dx, double dy, double charge, bool isSaturated, int plane, int digitIdx, int status);

  void removePad(size_t iPad);

  void sortPads(double precision);

  /// return the total number of pads associated to this cluster
  size_t multiplicity() const { return mPads.size(); }
  size_t multiplicity(int plane) const;

  /// return the ith pad (no bound checking)
  PadOriginal& pad(size_t i) { return mPads[i]; }
  const PadOriginal& pad(size_t i) const { return mPads[i]; }

  /// return begin/end iterators to be able to iterate over the pads without accessing the internal vector
  auto begin() { return mPads.begin(); }
  auto begin() const { return mPads.begin(); }
  auto end() { return mPads.end(); }
  auto end() const { return mPads.end(); }

  /// return the total charge of this cluster
  float charge() const { return mCharge[0] + mCharge[1]; }
  /// return the charge asymmetry of this cluster
  float chargeAsymmetry() const { return charge() > 0 ? TMath::Abs(mCharge[0] - mCharge[1]) / charge() : 0.; }
  /// return the plane with the highest charge
  int maxChargePlane() const { return mCharge[0] > mCharge[1] ? 0 : 1; }

  /// return whether there are saturated pads on *both* plane or not
  bool isSaturated() const { return mIsSaturated[0] && mIsSaturated[1]; }

  std::pair<double, double> minPadDimensions(int statusMask, bool matchMask) const;
  std::pair<double, double> minPadDimensions(int plane, int statusMask, bool matchMask) const;

  void area(int plane, double area[2][2]) const;

  std::pair<int, int> sizeInPads(int statusMask) const;
  std::pair<int, int> sizeInPads(int plane, int statusMask) const;

 private:
  std::vector<PadOriginal> mPads{};      ///< list of pads associated to this cluster
  size_t mMultiplicity[2] = {0, 0};      ///< number of pads in bending and non-bending planes
  float mCharge[2] = {0., 0.};           ///< integrated charge on both planes
  bool mIsSaturated[2] = {false, false}; ///< whether there are saturated pads on each plane
};

} // namespace mch
} // namespace o2

#endif // O2_MCH_CLUSTERORIGINAL_H_
