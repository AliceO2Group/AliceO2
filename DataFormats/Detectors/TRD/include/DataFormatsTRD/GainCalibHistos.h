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

/// \file GainCalibHistos.h
/// \brief Class to store the TRD dEdx distribution for each TRD chamber

#ifndef ALICEO2_GAINCALIBHISTOS_H
#define ALICEO2_GAINCALIBHISTOS_H

#include "DataFormatsTRD/Constants.h"
#include "Rtypes.h"
#include <array>
#include <gsl/span>

namespace o2
{
namespace trd
{

class GainCalibHistos
{
 public:
  GainCalibHistos() = default;
  GainCalibHistos(const GainCalibHistos&) = default;
  ~GainCalibHistos() = default;
  void reset();
  void addEntry(float dEdx, int chamberId);
  float getHistogramEntry(int index) const { return mdEdxEntries[index]; }
  size_t getNEntries() const { return mNEntriesTot; }

  void fill(const GainCalibHistos& input);
  void fill(const gsl::span<const GainCalibHistos> input); // dummy!
  void merge(const GainCalibHistos* prev);
  void print();

 private:
  std::array<int, constants::MAXCHAMBER * constants::NBINSGAINCALIB> mdEdxEntries{0}; ///< dEdx histograms
  size_t mNEntriesTot{0};

  ClassDefNV(GainCalibHistos, 1);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_GAINCALIBHISTOS_H
