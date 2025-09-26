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

/// \file Digit.h
/// \brief Definition of ECal digit class
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#ifndef ALICEO2_ECAL_DIGIT_H
#define ALICEO2_ECAL_DIGIT_H

#include <Rtypes.h>
#include <CommonDataFormat/TimeStamp.h>

namespace o2
{
namespace ecal
{
class Digit : public o2::dataformats::TimeStamp<double>
{
 public:
  Digit() = default;
  Digit(int tower, double amplitudeGeV, double time);
  ~Digit() = default;

  // setters
  void setTower(int tower) { mTower = tower; }
  void setAmplitude(double amplitude) { mAmplitudeGeV = amplitude; }
  void setEnergy(double energy) { mAmplitudeGeV = energy; }
  void setLabel(int label) { mLabel = label; }

  // getters
  int getTower() const { return mTower; }
  double getAmplitude() const { return mAmplitudeGeV; }
  double getEnergy() const { return mAmplitudeGeV; }
  int getLabel() const { return mLabel; }

 private:
  double mAmplitudeGeV = 0.; ///< Amplitude (GeV)
  int32_t mTower = -1;       ///< Tower index (absolute cell ID)
  int32_t mLabel = -1;       ///< Index of the corresponding entry/entries in the MC label array
  ClassDefNV(Digit, 1);
};

} // namespace ecal
} // namespace o2
#endif // ALICEO2_ECAL_DIGIT_H
