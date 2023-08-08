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

#ifndef O2_TRDTRAPCONFIGEVENT_H
#define O2_TRDTRAPCONFIGEVENT_H

#include "CommonDataFormat/InteractionRecord.h"
#include <fairlogger/Logger.h>

#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/MCMEvent.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/TrapRegisters.h"
#include "DataFormatsTRD/TrapRegInfo.h"

#include <string>
#include <map>
#include <unordered_map>
#include <array>
#include <vector>
#include <bitset>
#include <gsl/span>

namespace o2
{
namespace trd
{

class TrapConfigEvent
{
  // class that is actually stored in ccdb.
  // it holds a compressed version of what comes in the config events

 public:
  TrapConfigEvent();
  ~TrapConfigEvent() = default;

  TrapConfigEvent(const TrapConfigEvent& A);

  // get a config register value by index, addr, and name, via mcm index
  uint32_t getRegisterValue(const uint32_t regidx, const int mcmidx);
  bool setRegisterValue(const uint32_t data, const uint32_t regidx, const int mcmidx);

  // get a registers index (enum value) by address or name
  // due to the speed, it is never used in normal operations, only debugging.
  int32_t getRegIndexByName(const std::string& name);

  // get a registers address by index or name
  int32_t getRegAddrByIdx(const unsigned int regidx);
  int32_t getRegAddrByName(const std::string& name);

  // move to parser bool isValidAddress(const uint32_t addr);
  const std::string getRegisterName(const unsigned int regidx) { return mTrapRegisters[regidx].getName(); }
  const uint32_t getRegisterMax(const unsigned int regidx) { return mTrapRegisters[regidx].getMax(); }
  const uint32_t getRegisterNBits(const unsigned int regidx) { return mTrapRegisters[regidx].getNbits(); }
  const TrapRegInfo& getRegisterInfo(const uint32_t regidx) { return mTrapRegisters[regidx]; }
  const uint16_t getRegisterAddress(const uint32_t regidx) { return mTrapRegisters[regidx].getAddr(); }

  // return all the registers for a particular mcm
  void getAllRegisters(const int mcmidx, std::array<uint32_t, o2::trd::TrapRegisters::kLastReg>& registers);

  // return all the mcm values for a particular register index
  void getAllMCMByIndex(const int regindex, std::array<uint32_t, o2::trd::constants::MAXMCMCOUNT>& registers);

  // return all the mcm values for a particular register name
  void getAllMCMByName(const std::string& registername, std::array<uint32_t, o2::trd::constants::MAXMCMCOUNT>& mcms);

  // return all the config data in an unpacked array.
  void getAll(std::array<uint32_t, o2::trd::TrapRegisters::kLastReg * o2::trd::constants::MAXMCMCOUNT>& configdata);

  // population pending
  uint32_t getConfigVersion(const int mcmid = 0) { return getRegisterValue(TrapRegisters::kQ2VINFO, mcmid); }    // these must some how be gotten from git or wingdb.
  uint32_t getConfigName(const int mcmid = 0) { return getRegisterValue(TrapRegisters::kQ2VINFO, mcmid); }       // these must be gotten from git or wingdb.
  uint16_t getConfigSavedVersion(const int mcmid = 0) { return getRegisterValue(TrapRegisters::kVINFO, mcmid); } // the version that is saved, for the ability to later save the config differently.

  bool isConfigDifferent(const TrapConfigEvent& trapconfigevent) const;

  // for compliance with the same interface to o2::trd::TrapConfig and its run1/2 inherited interface:
  // only these 2 are used in the simulations.
  uint32_t getDmemUnsigned(uint32_t address, int detector, int rob, int mcm);
  uint32_t getTrapReg(const uint32_t index, const int detector, const int rob, const int mcm);

  //  bool isHCIDPresent(const int hcid) const { return mHCIDPresent.test(hcid); }
  //  void HCIDIsPresent(const int hcid) { mHCIDPresent.set(hcid); }
  //  uint32_t countHCIDPresent() { return mHCIDPresent.count(); }
  uint32_t countHCIDPresent() { return constants::MAXHALFCHAMBER - std::count(mHCIDPresentCount.begin(), mHCIDPresentCount.end(), -1); } // count those indices not set and subtract from the total number of mcm
                                                                                                                                         //  const std::bitset<constants::MAXHALFCHAMBER>& getHCIDPresent() const { return mHCIDPresent; }
  bool isMCMPresent(const int mcmid) const { return (mConfigDataIndex[mcmid] != -1) ? true : false; }
  // const std::bitset<constants::MAXMCMCOUNT>& getMCMPresent() const { return mMCMPresent; }
  uint32_t countMCMPresent() { return constants::MAXMCMCOUNT - std::count(mConfigDataIndex.begin(), mConfigDataIndex.end(), -1); } // count those indices not set and subtract from the total number of mcm
  // void clearMCMPresent() { mMCMPresent.reset(); }
  void clearMCMEvent() { mConfigData.clear(); }
  void clear()
  {
    // clearMCMPresent();
    clearMCMEvent();
  }
  bool ignoreWord(const int offset) const { return mWordNumberIgnore.test(offset); }

  // required for a container for calibration
  void fill(const TrapConfigEvent& input);
  void fill(const gsl::span<const TrapConfigEvent> input); // dummy!
  void merge(const TrapConfigEvent* prev);
  void print();
  int getRegisterBase(const int regidx) { return mTrapRegisters[regidx].getBase(); }
  const MCMEvent& getMCMEvent(const int mcmidx) { return mConfigData[mcmidx]; }
  int getMCMEventSize() { return mConfigDataIndex.size(); }
  uint32_t getrawdata(uint32_t idx) { return mConfigData[0].getvalue(idx); }
  // TODO put into QC
  void buildAverage();
  void buildDefaults();
  void setDefaultRegisterValue(const int regidx, const uint32_t registervalue) { mDefaultRegisters.setRegister(regidx, registervalue, mTrapRegisters[regidx]); }
  uint32_t getDefaultRegisterValue(const int regidx) { return mDefaultRegisters.getRegister(regidx, mTrapRegisters[regidx]); }

 private:
  TrapRegisters mTrapRegisters;
  MCMEvent mDefaultRegisters;                                           // default values for registers
  std::array<uint16_t, constants::MAXHALFCHAMBER> mHCIDPresentCount{0}; // did the link actually receive data.
  std::array<uint16_t, constants::MAXMCMCOUNT> mMCMPresentCount{0};     // how many times did this mcm receive a config event to build this config event. Ideally all present ones would have the same number, this is not the case and the reason for this array.
  std::vector<MCMEvent> mConfigData;                                    // vector of register data blocks
  std::array<int32_t, constants::MAXMCMCOUNT> mConfigDataIndex{-1};     // one block of data per mcm, array as one wants to query if an mcm is present with having to walk the whole index.
                                                                        //  std::map<uint16_t, uint16_t> mTrapRegistersAddressIndexMap;    moved to parser   //!< map of address into mTrapRegisters, populated at the end of initialiseRegisters
  std::bitset<kTrapRegistersSize> mWordNumberIgnore;                    // whether to ignore a register or not. Here to speed lookups up.
  // alternate storage:
  // std::array<std::vector<uint32_t>,o2::trd::TrapRegisters::kLastReg> mConfigDataint;                 // vector of vectors of register data, 1 value for a constant register, 69k for individual
  //  now we dont know if we actually have data for a specific mcm, so keep a record of which ones have been actually read and known to be current
  // std::bitset<o2::trd::constants::MAXMCMCOUNT> mMCMsPresent;                 // which mcms were actually seen to build this series of configuration events

  ClassDefNV(TrapConfigEvent, 5);
};

} // namespace trd
} // namespace o2

#endif
