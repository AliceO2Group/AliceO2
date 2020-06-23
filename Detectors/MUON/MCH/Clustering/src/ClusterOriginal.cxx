// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterOriginal.cxx
/// \brief Implementation of the cluster used by the original cluster finder algorithm
///
/// \author Philippe Pillot, Subatech

#include "ClusterOriginal.h"

#include <cassert>
#include <numeric>
#include <set>
#include <stdexcept>

#include "PadOriginal.h"

namespace o2
{
namespace mch
{

//_________________________________________________________________________________________________
bool shouldUsePad(const PadOriginal& pad, int plane, int statusMask, bool matchMask)
{
  /// return true if the pad should be used according to the plane it belongs to,
  /// its state and whether its status mask matches (or not) the given mask

  if (pad.plane() == plane && pad.isReal() && !pad.isSaturated()) {
    bool test = (statusMask != 0) ? ((pad.status() & statusMask) != 0) : (pad.status() == PadOriginal::kZero);
    if ((test && matchMask) || (!test && !matchMask)) {
      return true;
    }
  }

  return false;
}

//_________________________________________________________________________________________________
void ClusterOriginal::clear()
{
  /// clear the content of this cluster

  mPads.clear();

  mMultiplicity[0] = 0;
  mMultiplicity[1] = 0;

  mCharge[0] = 0.;
  mCharge[1] = 0.;

  mIsSaturated[0] = false;
  mIsSaturated[1] = false;
}

//_________________________________________________________________________________________________
void ClusterOriginal::addPad(double x, double y, double dx, double dy, double charge, bool isSaturated, int plane, int digitIdx, int status)
{
  /// add a new pad to this cluster

  assert(plane == 0 || plane == 1);

  mPads.emplace_back(x, y, dx, dy, charge, isSaturated, plane, digitIdx, status);

  ++mMultiplicity[plane];
  mCharge[plane] += charge;
  if (isSaturated) {
    mIsSaturated[plane] = true;
  }
}

//_________________________________________________________________________________________________
void ClusterOriginal::removePad(size_t iPad)
{
  /// remove the given pad from the internal list and update the cluster informations

  assert(iPad < mPads.size());

  mPads.erase(mPads.begin() + iPad);

  mMultiplicity[0] = 0;
  mMultiplicity[1] = 0;
  mCharge[0] = 0.;
  mCharge[1] = 0.;
  mIsSaturated[0] = false;
  mIsSaturated[1] = false;
  for (const auto& pad : mPads) {
    ++mMultiplicity[pad.plane()];
    mCharge[pad.plane()] += pad.charge();
    if (pad.isSaturated()) {
      mIsSaturated[pad.plane()] = true;
    }
  }
}

//_________________________________________________________________________________________________
void ClusterOriginal::sortPads(double precision)
{
  /// sort the pads per plane then in increasing y-position if same plane then in increasing x-position if same y
  /// positions within ± precision are considered as equal
  std::sort(mPads.begin(), mPads.end(), [precision](const PadOriginal& pad1, const PadOriginal& pad2) {
    return (pad1.plane() < pad2.plane() ||
            (pad1.plane() == pad2.plane() && (pad1.y() < pad2.y() - precision ||
                                              (pad1.y() < pad2.y() + precision && pad1.x() < pad2.x() - precision))));
  });
}

//_________________________________________________________________________________________________
size_t ClusterOriginal::multiplicity(int plane) const
{
  /// return the number of pads associated to this cluster, in total or in the given plane
  return (plane == 0 || plane == 1) ? mMultiplicity[plane] : mPads.size();
}

//_________________________________________________________________________________________________
PadOriginal& ClusterOriginal::pad(size_t i)
{
  /// return the ith pad (no bound checking)
  return mPads[i];
}

//_________________________________________________________________________________________________
std::pair<double, double> ClusterOriginal::minPadDimensions(int statusMask, bool matchMask) const
{
  /// Returns the minimum pad dimensions (half sizes), only considering
  /// pads matching (or not, depending matchMask) a given mask

  auto dim0(minPadDimensions(0, statusMask, matchMask));
  auto dim1(minPadDimensions(1, statusMask, matchMask));

  return std::make_pair(TMath::Min(dim0.first, dim1.first), TMath::Min(dim0.second, dim1.second));
}

//_________________________________________________________________________________________________
std::pair<double, double> ClusterOriginal::minPadDimensions(int plane, int statusMask, bool matchMask) const
{
  /// Returns the minimum pad dimensions (half sizes), only considering
  /// pads matching (or not, depending matchMask) a given mask, within a given plane

  assert(plane == 0 || plane == 1);

  double xmin(std::numeric_limits<float>::max());
  double ymin(std::numeric_limits<float>::max());

  if (mMultiplicity[plane] == 0) {
    return std::make_pair(xmin, ymin);
  }

  for (const auto& pad : mPads) {
    if (shouldUsePad(pad, plane, statusMask, matchMask)) {
      xmin = TMath::Min(xmin, pad.dx());
      ymin = TMath::Min(ymin, pad.dy());
    }
  }

  return std::make_pair(xmin, ymin);
}

//_________________________________________________________________________________________________
void ClusterOriginal::area(int plane, double area[2][2]) const
{
  /// return the geometrical area (cm) covered by the pads on the given plane
  /// area[0][0] = xmin, area[0][1] = xmax, area[1][0] = ymin, area[1][1] = ymax

  assert(plane == 0 || plane == 1);

  area[0][0] = std::numeric_limits<float>::max();
  area[0][1] = -std::numeric_limits<float>::max();
  area[1][0] = std::numeric_limits<float>::max();
  area[1][1] = -std::numeric_limits<float>::max();

  for (const auto& pad : mPads) {
    if (pad.plane() == plane) {
      area[0][0] = TMath::Min(area[0][0], pad.x() - pad.dx());
      area[0][1] = TMath::Max(area[0][1], pad.x() + pad.dx());
      area[1][0] = TMath::Min(area[1][0], pad.y() - pad.dy());
      area[1][1] = TMath::Max(area[1][1], pad.y() + pad.dy());
    }
  }
}

//_________________________________________________________________________________________________
std::pair<int, int> ClusterOriginal::sizeInPads(int statusMask) const
{
  /// return the size of the cluster in terms of number of pads in x and y directions
  /// use the pads from the plane in which their minimum size is the smallest in this direction

  std::pair<double, double> dim0 = minPadDimensions(0, statusMask, true);
  std::pair<double, double> dim1 = minPadDimensions(1, statusMask, true);

  std::pair<int, int> npad0 = sizeInPads(0, statusMask);
  std::pair<int, int> npad1 = sizeInPads(1, statusMask);

  int nx(0);
  if (TMath::Abs(dim0.first - dim1.first) < 1.e-3) {
    nx = TMath::Max(npad0.first, npad1.first);
  } else {
    nx = dim0.first < dim1.first ? npad0.first : npad1.first;
  }

  int ny(0);
  if (TMath::Abs(dim0.second - dim1.second) < 1.e-3) {
    ny = TMath::Max(npad0.second, npad1.second);
  } else {
    ny = dim0.second < dim1.second ? npad0.second : npad1.second;
  }

  return std::make_pair(nx, ny);
}

//_________________________________________________________________________________________________
std::pair<int, int> ClusterOriginal::sizeInPads(int plane, int statusMask) const
{
  /// return the size of the cluster on the given plane in terms of number of pads in x and y directions

  assert(plane == 0 || plane == 1);

  if (mMultiplicity[plane] == 0) {
    return std::make_pair(0, 0);
  }

  // order pads in x and y directions considering positions closer than 0.01 cm as equal
  auto cmp = [](double a, double b) { return a < b - 0.01; };
  std::set<double, decltype(cmp)> padx(cmp);
  std::set<double, decltype(cmp)> pady(cmp);

  for (const auto& pad : mPads) {
    if (shouldUsePad(pad, plane, statusMask, true)) {
      padx.emplace(pad.x());
      pady.emplace(pad.y());
    }
  }

  return std::make_pair(padx.size(), pady.size());
}

} // namespace mch
} // namespace o2
