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

#include "Framework/Logger.h"
#include "CommonDataFormat/BunchFilling.h"
#include <TClass.h>
#include <TFile.h>
#include <cctype>

using namespace o2;

//_________________________________________________
BunchFilling::BunchFilling(const std::string& beamA, const std::string& beamC)
{
  setBCFilling(beamA, 0);
  setBCFilling(beamC, 1);
  setInteractingBCsFromBeams();
}

//_________________________________________________
BunchFilling::BunchFilling(const std::string& interactingBC)
{
  setBCFilling(interactingBC, -1);
  mBeamAC[0] = mBeamAC[1] = mPattern;
}

//_________________________________________________
int BunchFilling::getFirstFilledBC(int dir) const
{
  for (int bc = 0; bc < o2::constants::lhc::LHCMaxBunches; bc++) {
    if (testBC(bc, dir)) {
      return bc;
    }
  }
  return -1;
}

//_________________________________________________
int BunchFilling::getLastFilledBC(int dir) const
{
  for (int bc = o2::constants::lhc::LHCMaxBunches; bc--;) {
    if (testBC(bc, dir)) {
      return bc;
    }
  }
  return -1;
}

//_________________________________________________
std::vector<int> BunchFilling::getFilledBCs(int dir) const
{
  std::vector<int> vb;
  for (int bc = 0; bc < o2::constants::lhc::LHCMaxBunches; bc++) {
    if (testBC(bc, dir)) {
      vb.push_back(bc);
    }
  }
  return vb;
}

//_________________________________________________
void BunchFilling::setBC(int bcID, bool active, int dir)
{
  // add interacting BC slot
  if (bcID >= o2::constants::lhc::LHCMaxBunches) {
    LOG(fatal) << "BCid is limited to " << 0 << '-' << o2::constants::lhc::LHCMaxBunches - 1;
  }
  if (dir < 0) {
    mPattern.set(bcID, active);
  } else {
    mBeamAC[dir].set(bcID, active);
  }
}

//_________________________________________________
void BunchFilling::mergeWith(o2::BunchFilling const& other)
{
  for (int bc = o2::constants::lhc::LHCMaxBunches; bc--;) {
    for (int dir = -1; dir < 2; dir++) {
      if (other.testBC(bc, dir)) {
        // if the other one is filled we need to fill "this" also
        setBC(bc, true, dir);
      }
    }
  }
}

//_________________________________________________
void BunchFilling::setBCTrain(int nBC, int bcSpacing, int firstBC, int dir)
{
  // add interacting BC train with given spacing starting at given place, i.e.
  // train with 25ns spacing should have bcSpacing = 1
  for (int i = 0; i < nBC; i++) {
    setBC(firstBC, dir);
    firstBC += bcSpacing;
  }
}

//_________________________________________________
void BunchFilling::setBCTrains(int nTrains, int trainSpacingInBC, int nBC, int bcSpacing, int firstBC, int dir)
{
  // add nTrains trains of interacting BCs with bcSpacing within the train and trainSpacingInBC empty slots
  // between the trains
  for (int it = 0; it < nTrains; it++) {
    setBCTrain(nBC, bcSpacing, firstBC, dir);
    firstBC += nBC * bcSpacing + trainSpacingInBC;
  }
}

//_________________________________________________
void BunchFilling::setBCFilling(const std::string& patt, int dir)
{
  auto& dest = dir < 0 ? mPattern : mBeamAC[dir];
  dest = createPattern(patt);
}

//_________________________________________________
void BunchFilling::print(int dir, bool filledOnly, int bcPerLine) const
{
  const std::string names[3] = {"Interacting", "Beam-A", "Beam-C"};
  for (int id = -1; id < 2; id++) {
    if (id == dir || (dir < -1 || dir > 1)) {
      printf("%s bunches\n", names[id + 1].c_str());
      bool endlOK = false;
      for (int i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
        bool on = testBC(i, id);
        if (!filledOnly) {
          printf("%c", on ? '+' : '-');
        } else if (on) {
          printf("%4d ", i);
        } else {
          continue;
        }
        if (((i + 1) % bcPerLine) == 0) {
          printf("\n");
          endlOK = true;
        } else {
          endlOK = false;
        }
      }
      if (!endlOK) {
        printf("\n");
      }
    }
  }
}

//_______________________________________________
BunchFilling* BunchFilling::loadFrom(const std::string& fileName, const std::string& objName)
{
  // load object from file
  TFile fl(fileName.data());
  if (fl.IsZombie()) {
    LOG(error) << "Failed to open " << fileName;
    return nullptr;
  }
  std::string nm = objName.empty() ? o2::BunchFilling::Class()->GetName() : objName;
  auto bf = reinterpret_cast<o2::BunchFilling*>(fl.GetObjectChecked(nm.c_str(), o2::BunchFilling::Class()));
  if (!bf) {
    LOG(error) << "Did not find object named " << nm;
    return nullptr;
  }
  return bf;
}

//_________________________________________________
BunchFilling::Pattern BunchFilling::createPattern(const std::string& p)
{
  // create bit pattern from string according to convention of Run1/2:
  // The string has the following syntax:
  // "25L 25(2H2LH 3(23HL))"
  // - H/h -> 1  L/l -> 0
  // - spaces, new lines are white characters
  std::vector<unsigned char> input;
  for (unsigned char c : p) {
    if (c == '(' || c == ')' || std::isdigit(c)) {
      input.push_back(c);
    } else if (c == 'h' || c == 'H') {
      input.push_back('H');
    } else if (c == 'l' || c == 'L') {
      input.push_back('L');
    } else if (!std::isspace(c)) {
      throw std::runtime_error(fmt::format("forbidden character ({}) in input string ({})", c, p));
    }
  }
  input.push_back(0); // flag the end
  BunchFilling::Pattern patt;
  int ibit = 0, level = 0;
  const unsigned char* in = input.data();
  if (!parsePattern(in, patt, ibit, level)) {
    throw std::runtime_error(fmt::format("failed to extract BC pattern from input string ({})", p));
  }
  return patt;
}

//_________________________________________________
std::string BunchFilling::buckets2PatternString(const std::vector<int>& buckets, int ibeam)
{
  // create bunches pattern string from filled buckets, in the format parsed by o2::BunchFilling::createPattern
  Pattern patt;
  std::string pattS;
  for (int b : buckets) {
    patt[o2::constants::lhc::LHCBunch2P2BC(b / 10, o2::constants::lhc::BeamDirection(ibeam))] = true;
  }
  int nh = 0;
  for (int i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
    if (patt[i]) {
      if (nh) {
        pattS += fmt::format("{}L", nh);
        nh = 0; // reset holes
      }
      pattS += "H";
    } else {
      nh++;
    }
  }
  return pattS;
}

//_________________________________________________
void BunchFilling::buckets2BeamPattern(const std::vector<int>& buckets, int ibeam)
{
  // create bunches pattern string from filled buckets, in the format parsed by o2::BunchFilling::createPattern
  Pattern& patt = mBeamAC[ibeam];
  for (int b : buckets) {
    if (b) {
      patt[o2::constants::lhc::LHCBunch2P2BC(b / 10, o2::constants::lhc::BeamDirection(ibeam))] = true;
    }
  }
}

//_________________________________________________
bool BunchFilling::parsePattern(const unsigned char*& input, BunchFilling::Pattern& patt, int& ibit, int& level)
{
  // this is analog of AliTriggerBCMask::Bcm2Bits
  level++;
  int repetion = 1;
  auto setBunch = [&patt, &ibit](bool v) {
    if (ibit < patt.size()) {
      patt[ibit++] = v;
    } else {
      throw std::runtime_error(fmt::format("provided pattern overflow max bunch {}", patt.size()));
    }
  };
  while (1) {
    if (*input == 0) {
      if (level > 1) {
        LOG(error) << "Missing )";
        return false;
      }
      break;
    }
    if (*input == 'H') {
      for (int i = 0; i < repetion; i++) {
        setBunch(true);
      }
      repetion = 1;
      input++;
    } else if (*input == 'L') {
      for (int i = 0; i < repetion; i++) {
        setBunch(false);
      }
      repetion = 1;
      input++;
    } else if (std::isdigit(*input)) {
      repetion = int((*input++) - '0');
      while (*input && std::isdigit(*input)) {
        repetion = repetion * 10 + int(*(input++) - '0');
      }
    } else if (*input == '(') {
      input++;
      auto ibit1 = ibit;
      if (!parsePattern(input, patt, ibit, level)) {
        return false;
      }
      auto ibit2 = ibit;
      for (int i = 0; i < repetion - 1; i++) {
        for (int j = ibit1; j < ibit2; j++) {
          setBunch(patt[j]);
        }
      }
      repetion = 1;
    } else if (*input == ')') {
      input++;
      if (level <= 1) {
        LOG(error) << "Incorrectly placed )";
        return false;
      }
      break;
    } else {
      LOG(error) << "Unexpected symbol " << *input;
      return false;
    }
  }
  level--;
  return true;
}
