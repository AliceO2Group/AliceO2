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

/// @brief pattern of filled (interacting) bunches

#ifndef ALICEO2_BUNCHFILLING_H
#define ALICEO2_BUNCHFILLING_H

#include "CommonConstants/LHCConstants.h"
#include <bitset>
#include <string>

namespace o2
{
class BunchFilling
{
 public:
  int getNBunches() const { return mPattern.count(); }
  bool testBC(int bcID) const { return mPattern[bcID]; }
  void setBC(int bcID, bool active = true);
  void setBCTrain(int nBC, int bcSpacing, int firstBC);
  void setBCTrains(int nTrains, int trainSpacingInBC, int nBC, int bcSpacing, int firstBC);
  void print(int bcPerLine = 100) const;
  const auto& getPattern() const { return mPattern; }
  int getFirstFilledBC() const;
  int getLastFilledBC() const;
  // set BC filling a la TPC TDR, 12 50ns trains of 48 BCs
  // but instead of uniform train spacing we add 96empty BCs after each train
  void setDefault()
  {
    setBCTrains(12, 96, 48, 2, 0);
  }

  static BunchFilling* loadFrom(const std::string& fileName, const std::string& objName = "");

 private:
  std::bitset<o2::constants::lhc::LHCMaxBunches> mPattern;

  ClassDefNV(BunchFilling, 1);
};
} // namespace o2

#endif
