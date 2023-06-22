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
#include "Framework/InputRecord.h"
#include "Rtypes.h"
#include <vector>
#include <memory>
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
  void init();
  void addEntry(float dEdx, int chamberId);
  auto getHistogramEntry(int index) const { return mdEdxEntries[index]; }
  auto getNEntries() const { return mNEntriesTot; }

  void fill(const std::unique_ptr<const GainCalibHistos, o2::framework::InputRecord::Deleter<const o2::trd::GainCalibHistos>>& input);
  void merge(const GainCalibHistos* prev);
  void print();

 private:
  std::vector<int> mdEdxEntries{}; ///< dEdx histograms
  size_t mNEntriesTot{0};
  bool mInitialized{false};

  ClassDefNV(GainCalibHistos, 1);
};

} // namespace trd
} // namespace o2

#endif // ALICEO2_GAINCALIBHISTOS_H
