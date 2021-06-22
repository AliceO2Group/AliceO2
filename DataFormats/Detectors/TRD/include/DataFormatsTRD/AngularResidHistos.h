// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AngularResidHistos.h
/// \brief Class to store the angular residuals of TRD tracklets wrt TPC tracks for each TRD chamber

#ifndef ALICEO2_ANGRESIDHISTOS_H
#define ALICEO2_ANGRESIDHISTOS_H

#include "DataFormatsTRD/Constants.h"
#include "Rtypes.h"
#include <array>
#include <gsl/span>

namespace o2
{
namespace trd
{

class AngularResidHistos
{
 public:
  AngularResidHistos() = default;
  AngularResidHistos(const AngularResidHistos&) = default;
  ~AngularResidHistos() = default;
  bool addEntry(float deltaAlpha, float impactAngle, int chamberId);
  float getHistogramEntry(int index) const { return mHistogramEntries[index]; }
  int getBinCount(int index) const { return mNEntriesPerBin[index]; }
  size_t getNEntries() const { return mNEntriesTotal; }

  void fill(const gsl::span<const AngularResidHistos> input);
  void merge(const AngularResidHistos* prev);
  void print();

 private:
  // TODO use NCHAMBER instead of MAXCHAMBER and indirection array?
  static constexpr float INVBINWIDTH = constants::NBINSANGLEDIFF / (2.f * constants::MAXIMPACTANGLE);
  std::array<float, constants::MAXCHAMBER * constants::NBINSANGLEDIFF> mHistogramEntries{}; ///< sum of angular deviation (tracklet to track) for given track angle
  std::array<int, constants::MAXCHAMBER * constants::NBINSANGLEDIFF> mNEntriesPerBin{};     ///< number of entries per bin (needed for calculation of mean)
  size_t mNEntriesTotal{0};                                                                 ///< total number of accumulated angular deviations

  ClassDefNV(AngularResidHistos, 1);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_ANGRESIDHISTOS_H
