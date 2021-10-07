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

  std::pair<long, int32_t> getBeamEnergy() const { return mBeamEnergy; }
  std::pair<long, int32_t> getFillNumber() const { return mFillNumber; }
  std::pair<long, std::string> getInjectionScheme() const { return mInjectionScheme; }
  std::pair<long, int32_t> getAtomicNumberB1() const { return mAtomicNumberB1; }
  std::pair<long, int32_t> getAtomicNumberB2() const { return mAtomicNumberB2; }

  int32_t getBeamEnergyVal() const { return mBeamEnergy.second; }
  int32_t getFillNumberVal() const { return mFillNumber.second; }
  std::string getInjectionSchemeVal() const { return mInjectionScheme.second; }
  int32_t getAtomicNumberB1Val() const { return mAtomicNumberB1.second; }
  int32_t getAtomicNumberB2Val() const { return mAtomicNumberB2.second; }

  long getBeamEnergyTime() const { return mBeamEnergy.first; }
  long getFillNumberTime() const { return mFillNumber.first; }
  long getInjectionSchemeTime() const { return mInjectionScheme.first; }
  long getAtomicNumberB1Time() const { return mAtomicNumberB1.first; }
  long getAtomicNumberB2Time() const { return mAtomicNumberB2.first; }

  void setBeamEnergy(std::pair<long, int32_t> p) { mBeamEnergy = p; }
  void setFillNumber(std::pair<long, int32_t> p) { mFillNumber = p; }
  void setInjectionScheme(std::pair<long, std::string> p) { mInjectionScheme = p; }
  void setAtomicNumberB1(std::pair<long, int32_t> p) { mAtomicNumberB1 = p; }
  void setAtomicNumberB2(std::pair<long, int32_t> p) { mAtomicNumberB2 = p; }

  void setBeamEnergy(long t, int32_t v) { mBeamEnergy = std::make_pair(t, v); }
  void setFillNumber(long t, int32_t v) { mFillNumber = std::make_pair(t, v); }
  void setInjectionScheme(long t, std::string v) { mInjectionScheme = std::make_pair(t, v); }
  void setAtomicNumberB1(long t, int32_t v) { mAtomicNumberB1 = std::make_pair(t, v); }
  void setAtomicNumberB2(long t, int32_t v) { mAtomicNumberB2 = std::make_pair(t, v); }

 private:
  std::pair<long, int32_t> mBeamEnergy;
  std::pair<long, int32_t> mFillNumber;
  std::pair<long, std::string> mInjectionScheme;
  std::pair<long, int32_t> mAtomicNumberB1;
  std::pair<long, int32_t> mAtomicNumberB2;

  ClassDefNV(LHCIFData, 1);
};
} // namespace parameters
} // namespace o2
#endif
