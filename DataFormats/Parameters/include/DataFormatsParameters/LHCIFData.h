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

/// \file LHCIFData.h
/// \brief container for the LHC InterFace data

#ifndef O2_GRP_LHCIFDATA_H
#define O2_GRP_LHCIFDATA_H

#include <Rtypes.h>
#include <string>
#include <cstdint>

namespace o2
{
namespace parameters
{

class LHCIFData
{
 public:
  LHCIFData() = default;
  ~LHCIFData() = default;

  std::pair<float, int32_t> getBeamEnergy() const { return mBeamEnergy; }
  std::pair<float, std::string> getFillNumber() const { return mFillNumber; }
  std::pair<float, std::string> getInjectionScheme() const { return mInjectionScheme; }
  std::pair<float, int32_t> getAtomicNumberB1() const { return mAtomicNumberB1; }
  std::pair<float, int32_t> getAtomicNumberB2() const { return mAtomicNumberB2; }

  int32_t getBeamEnergyVal() const { return mBeamEnergy.second; }
  std::string getFillNumberVal() const { return mFillNumber.second; }
  std::string getInjectionSchemeVal() const { return mInjectionScheme.second; }
  int32_t getAtomicNumberB1Val() const { return mAtomicNumberB1.second; }
  int32_t getAtomicNumberB2Val() const { return mAtomicNumberB2.second; }

  float getBeamEnergyTime() const { return mBeamEnergy.first; }
  float getFillNumberTime() const { return mFillNumber.first; }
  float getInjectionSchemeTime() const { return mInjectionScheme.first; }
  float getAtomicNumberB1Time() const { return mAtomicNumberB1.first; }
  float getAtomicNumberB2Time() const { return mAtomicNumberB2.first; }

  void setBeamEnergy(std::pair<float, int32_t> p) { mBeamEnergy = p; }
  void setFillNumber(std::pair<float, std::string> p) { mFillNumber = p; }
  void setInjectionScheme(std::pair<float, std::string> p) { mInjectionScheme = p; }
  void setAtomicNumberB1(std::pair<float, int32_t> p) { mAtomicNumberB1 = p; }
  void setAtomicNumberB2(std::pair<float, int32_t> p) { mAtomicNumberB2 = p; }

  void setBeamEnergy(float t, int32_t v) { mBeamEnergy = std::make_pair(t, v); }
  void setFillNumber(float t, std::string v) { mFillNumber = std::make_pair(t, v); }
  void setInjectionScheme(float t, std::string v) { mInjectionScheme = std::make_pair(t, v); }
  void setAtomicNumberB1(float t, int32_t v) { mAtomicNumberB1 = std::make_pair(t, v); }
  void setAtomicNumberB2(float t, int32_t v) { mAtomicNumberB2 = std::make_pair(t, v); }

 private:
  std::pair<float, int32_t> mBeamEnergy;
  std::pair<float, std::string> mFillNumber;
  std::pair<float, std::string> mInjectionScheme;
  std::pair<float, int32_t> mAtomicNumberB1;
  std::pair<float, int32_t> mAtomicNumberB2;

  ClassDefNV(LHCIFData, 1);
};
} // namespace parameters
} // namespace o2
#endif
