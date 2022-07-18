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

#ifndef ALICEO2_EMCAL_FEEDCS_H_
#define ALICEO2_EMCAL_FEEDCS_H_

#include <memory>
#include <iosfwd>
#include <array>
#include <Rtypes.h>
#include <bitset>

#include "EMCALCalib/TriggerDCS.h"

namespace o2
{

namespace emcal
{

const Int_t kNTRU = 46; // this

class FeeDCS
{

 public:
  /// \brief default constructor
  FeeDCS() = default;

  /// \brief Destructor
  ~FeeDCS() = default;

  /// \brief copy constructor
  FeeDCS(const FeeDCS& fee) = default;

  /// \brief Assignment operator
  FeeDCS& operator=(const FeeDCS& source) = default;

  bool operator==(const FeeDCS& other) const;

  o2::emcal::TriggerDCS getTriggerDCS() const { return mTrigDCS; }
  o2::emcal::TriggerTRUDCS getTRUDCS(Int_t iTRU) const { return mTrigDCS.getTRUDCS(iTRU); }
  o2::emcal::TriggerSTUDCS getSTUDCSEMCal() const { return mTrigDCS.getSTUDCSEMCal(); }
  o2::emcal::TriggerSTUDCS getSTUDCSDCal() const { return mTrigDCS.getSTUDCSDCal(); }
  std::bitset<32> getDDLlist0() const { return mLinks0; }
  std::bitset<14> getDDLlist1() const { return mLinks1; }
  unsigned int getSRUFWversion(int ism = 0) const { return mSRUFWversion.at(ism); }
  unsigned int getSRUconfig(int ism = 0) const { return mSRUcfg.at(ism); }
  int getNSRUbuffers(int ism = 0) const { return (mSRUcfg.at(ism) >> 1 & 0x7); }
  int getRunNumber() const { return mRunNumber; }

  void setTRUDCS(Int_t iTRU, o2::emcal::TriggerTRUDCS tru) { mTrigDCS.setTRU(iTRU, tru); }
  void setSTUEMCal(o2::emcal::TriggerSTUDCS stu) { mTrigDCS.setSTUEMCal(stu); }
  void setSTUDCal(o2::emcal::TriggerSTUDCS stu) { mTrigDCS.setSTUDCal(stu); }
  void setDDLlist0(unsigned int a) { mLinks0 = std::bitset<32>(a); }
  void setDDLlist1(unsigned int a) { mLinks1 = std::bitset<14>(a); }
  void setSRUFWversion(int ism, unsigned int ver) { mSRUFWversion.at(ism) = ver; }
  void setSRUconfig(int ism, unsigned int ver) { mSRUcfg.at(ism) = ver; }
  void setRunNumber(int rn) { mRunNumber = rn; }

  bool isDDLactive(int iDDL) { return (iDDL < 32 ? mLinks0.test(iDDL) : mLinks1.test(iDDL - 32)); }
  bool isSMactive(int iSM);
  bool isEMCalSTUactive() { return mLinks1.test(12); } // EMCAL STU FEEID is 44
  bool isDCalSTUactive() { return mLinks1.test(13); }  // EMCAL STU FEEID is 45

 private:
  int mRunNumber = 0;                         ///< Run Number
  o2::emcal::TriggerDCS mTrigDCS;             ///< TRU and STU config
  std::bitset<32> mLinks0;                    ///< info on first 32 DDLs included in RO, 1 means the DDL is active
  std::bitset<14> mLinks1;                    ///< info on remining 14 DDLs included in RO, 1 means the DDL is active
  std::array<unsigned int, 20> mSRUFWversion; ///< SRU FW version
  std::array<unsigned int, 20> mSRUcfg;       ///< SRU configuration

  ClassDefNV(FeeDCS, 1);
};

std::ostream& operator<<(std::ostream& in, const FeeDCS& dcs);

} // namespace emcal

} // namespace o2
#endif
