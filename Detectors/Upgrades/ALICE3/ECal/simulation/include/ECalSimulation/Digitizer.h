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

/// \file Digitizer.h
/// \brief Digitization of ECal MC information
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#ifndef ALICEO2_ECAL_DIGITIZER_H
#define ALICEO2_ECAL_DIGITIZER_H
#include <vector>
#include <ECalBase/Hit.h>
#include <DataFormatsECal/Digit.h>
#include <DataFormatsECal/MCLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>

using o2::ecal::Digit;
using o2::ecal::Hit;
using o2::ecal::MCLabel;

namespace o2
{
namespace ecal
{
class Digitizer
{
 public:
  Digitizer();
  ~Digitizer() = default;
  Digitizer(const Digitizer&) = delete;
  Digitizer& operator=(const Digitizer&) = delete;
  void init() {}
  void finish() {}
  void processHits(const std::vector<Hit>* mHits, std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<MCLabel>& labels, int collId);
  void setThreshold(double threshold) { mThreshold = threshold; }
  void setSmearCrystal(bool smearCrystal) { mSmearCrystal = smearCrystal; }
  void setSamplingFraction(double fraction) { mSamplingFraction = fraction; }
  void setCrystalPePerGeV(double pePerGeV) { mCrystalPePerGeV = pePerGeV; }

 private:
  std::vector<Digit> mArrayD;
  bool mSmearCrystal = 0;
  double mThreshold = 0.001;
  double mSamplingFraction = 9.8;
  double mCrystalPePerGeV = 4000;
};
} // namespace ecal
} // namespace o2

#endif // ALICEO2_ECAL_DIGITIZER_H
