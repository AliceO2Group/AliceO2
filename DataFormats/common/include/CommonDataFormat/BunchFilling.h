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
#include <Rtypes.h>
#include <bitset>
#include <string>
#include <array>

namespace o2
{
class BunchFilling
{
 public:
  using Pattern = std::bitset<o2::constants::lhc::LHCMaxBunches>;

  BunchFilling() = default;
  BunchFilling(const std::string& beamA, const std::string& beamC);
  BunchFilling(const std::string& interactingBC);

  // this is a pattern creator similar to Run1/2 AliTriggerBCMask
  // The string has the following syntax:
  // "25L 25(2H2LH 3(23HL))"
  //  - H/h -> 1  L/l -> 0
  //  - spaces, new lines are white characters
  static Pattern createPattern(const std::string& p);

  // create pattern string from filled bucket
  static std::string buckets2PatternString(const std::vector<int>& buckets, int ibeam);

  // get interacting bunches pattern (B)
  const auto& getBCPattern() const { return mPattern; }

  // get pattern or clockwise (0, A) and anticlockwise (1, C) beams at P2
  const auto& getBeamPattern(int beam) const { return mBeamAC[beam]; }

  // get pattern of interacting BCs (-1) or beams filled BCs at P2 (0,1)
  const auto& getPattern(int dir = -1) const { return dir < 0 ? getBCPattern() : getBeamPattern(dir); }

  // create pattern from filled bucket
  void buckets2BeamPattern(const std::vector<int>& buckets, int ibeam);

  // get number of interacting bunches (-1) and number of filled bunches for clockwise (0, A) and anticlockwise (1, C) beams
  int getNBunches(int dir = -1) const { return dir < 0 ? mPattern.count() : mBeamAC[dir].count(); }

  // test interacting bunch
  bool testInteractingBC(int bcID) const { return mPattern[bcID]; }

  // test bean bunch
  bool testBeamBunch(int bcID, int dir) const { return mBeamAC[dir][bcID]; }

  // test interacting (-1) or clockwise (0, A) and anticlockwise (1, C) beams bunch
  bool testBC(int bcID, int dir = -1) const { return dir < 0 ? testInteractingBC(bcID) : testBeamBunch(bcID, dir); }

  // BC setters, dir=-1 is for interacting bunches pattern, 0, 1 for clockwise (C) and anticlockwise (A) beams
  void setBC(int bcID, bool active = true, int dir = -1);
  void setBCTrain(int nBC, int bcSpacing, int firstBC, int dir = -1);
  void setBCTrains(int nTrains, int trainSpacingInBC, int nBC, int bcSpacing, int firstBC, int dir = -1);

  // new format for setting bunches pattern, see createPattern comments
  void setBCFilling(const std::string& patt, int dir = -1);

  void setInteractingBCsFromBeams() { mPattern = getBeamPattern(0) & getBeamPattern(1); }

  int getFirstFilledBC(int dir = -1) const;
  int getLastFilledBC(int dir = -1) const;

  // print pattern of bunches, dir=0,1: for C,A beams, dir=-1: for interacting BCs, otherwise: all
  void print(int dir = -2, bool filledOnly = true, int bcPerLine = 20) const;

  // get vector with filled BCs
  std::vector<int> getFilledBCs(int dir = -1) const;

  // set BC filling a la TPC TDR, 12 50ns trains of 48 BCs
  // but instead of uniform train spacing we add 96empty BCs after each train
  void setDefault()
  {
    //    setBCTrains(12, 96, 48, 2, 0); // obsolete way of setting the trains
    setBCFilling("12(48(HL) 96L)", 0);
    setBCFilling("12(48(HL) 96L)", 1);
    setInteractingBCsFromBeams();
  }

  // merge this bunch filling with other
  void mergeWith(o2::BunchFilling const& other);

  static BunchFilling* loadFrom(const std::string& fileName, const std::string& objName = "");

 private:
  static bool parsePattern(const unsigned char*& input, Pattern& patt, int& ibit, int& level);

  Pattern mPattern{};                                                 // Pattern of interacting BCs at P2
  std::array<Pattern, o2::constants::lhc::NBeamDirections> mBeamAC{}; // pattern of 2 beam bunches at P2, 0 for A, 1 for C beam

  ClassDefNV(BunchFilling, 2);
};
} // namespace o2

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::BunchFilling> : std::true_type {
};

} // namespace framework

#endif
