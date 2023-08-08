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

#ifndef O2_TRDTRAPCONFIGPARSER_H
#define O2_TRDTRAPCONFIGPARSER_H

#include <fairlogger/Logger.h>
#include "CommonDataFormat/InteractionRecord.h"

#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/TrapConfigEvent.h"
#include "DataFormatsTRD/TrapConfigEventQC.h"
#include "DataFormatsTRD/RawData.h"

#include <chrono>  // chrono::system_clock
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <string>  // string
#include <vector>
#include <memory>
#include <map>
#include <tuple>
#include <bitset>
#include <gsl/span>
#include "Rtypes.h"
#include "TH2F.h"

// Configuration of the TRD Tracklet Processor
// (TRD Front-End Electronics)
// There is a manual to describe all the internals of the electronics.
// A very detailed manual, will try put it in the source code.
// TRAP registers
// TRAP data memory (DMEM)
//

namespace o2
{
namespace trd
{

enum TrapConfigEventErrorTypes {
  kTrapConfigEventAllGood = 0x1,
  kTrapConfigEventNoEnd = 0x2,
  kTrapConfigEventNoBegin = 0x4,
  kTrapConfigEventNoGarbageEnd = 0x8,
  kTrapConfigEventNoGarbageBegin = 0x10,
  kTrapConfigEventLastError = 0x20,
  kTrapConfigEventRegisterGap = 0x40,
  kTrapConfigEventCorruptData = 0x80
}; // possible errors for parsing.

class TrapConfigEventParser
{
  // class used to unpack the trap config events, and figure out what to do to them.
 public:
  TrapConfigEventParser();
  ~TrapConfigEventParser();

  // int getTrapReg(TrapReg reg, int det = -1, int rob = -1, int mcm = -1);

  unsigned int getDmemUnsigned(int addr, int det, int rob, int mcm);

  // TrapReg getRegByAddress(int address);

  void printMCMRegisterCount(int hcid);
  void unpackBlockHeader(uint32_t& header, uint32_t& registerdata, uint32_t& step, uint32_t& bwidth, uint32_t& nwords, uint32_t& registeraddr, uint32_t& exit_flag);
  bool parse(std::vector<uint32_t>& data);
  int parseLink(std::vector<uint32_t>& data, uint32_t start, uint32_t end);
  int parseSingleData(std::vector<uint32_t>& data, uint32_t header, uint32_t& idx, uint32_t end, bool& fastforward);
  int parseBlockData(std::vector<uint32_t>& data, uint32_t header, uint32_t& idx, uint32_t end, bool& fastforward);

  bool checkRegister(uint32_t& registeraddr, uint32_t& registerdata);

  void FillHistograms(int eventnum); //;TH2F *hists[6])
  void compareToTrackletsHCID(std::bitset<1080> trackletshcid);
  std::array<int, TrapRegisters::kLastReg>& getStartRegArray() { return mStartReg; }      // the number of time this register was read as the first register
  std::array<int, TrapRegisters::kLastReg>& getStopRegArray() { return mStopReg; }        // the number of time this register was read as the last register
  std::array<int, TrapRegisters::kLastReg>& getMissedRegArray() { return mMissedReg; }    // the number of times this register was not read
  std::array<int, TrapRegisters::kLastReg>& getRegisterCount() { return mRegisterCount; } // total count for each register
  void init(){};
  TrapConfigEvent getNewConfig() { return *(mTrapConfigEvent.get()); };
  TrapConfigEvent* getNewConfigPtr() { return mTrapConfigEvent.get(); };
  // TrapConfigEvent getNewConfig() { return *mTrapConfigEvent.get(); };
  // TrapConfigEvent* getNewConfigPtr() { return mTrapConfigEvent.get(); };
  void sendTrapConfigEvent(framework::ProcessingContext& pc);

  uint32_t countHCIDPresent() const { return mHCHasBeenSeen.count(); }
  uint32_t countMCMPresent() const { return mMCMHasBeenSeen.count(); }

  // flush the stats that are config event specific
  // a single parse is almost certainly not going to contain the complete event, this is then run after we think we have the complete set.
  int flushParsingStats();
  int64_t getConfigSize() { return sizeof(*mTrapConfigEvent.get()); }
  void setMCMParsingStatus(uint32_t mcmid, int status) { mMcmParsingStatus[mcmid] |= status; }
  int getMCMParsingStatus(uint32_t mcmid) { return mMcmParsingStatus[mcmid]; }

  bool setRegister(const uint32_t regidx, const uint32_t mcmid, const uint32_t registerdata);
  const uint32_t getRegister(const uint32_t regidx, const uint32_t mcmid);
  void addMCM(const int mcmid);
  void clearEventBasedStats();
  void analyseEventBaseStats();
  void buildAddressMap();
  const int32_t getRegIndexByAddr(unsigned int addr);
  bool isValidAddress(uint32_t addr);
  const std::string getRegNameByAddr(uint16_t addr);
  const std::string getRegNameByIdx(const uint32_t regidx) { return mTrapConfigEvent.get()->getRegisterName(regidx); }
  void getRegisterByAddr(uint32_t registeraddr, std::string& regname, int32_t& newregidx, int32_t& numberbits);

 private:
  uint32_t mCurrentHCID = 0;
  DigitMCMHeader mCurrentMCMHeader;
  uint32_t mCurrentMCMID = 0;
  uint32_t mCurrentDataIndex = 0;
  uint32_t mCurrentHCTime = 0;
  uint32_t mTrapConfigEventCounter = 0;
  uint32_t mLastRegIndex = 0;
  uint32_t mRegistersReadForCurrentMCM = 0;
  uint32_t mCurrentRegisterWordsCount = 0;
  uint32_t mPreviousRegisterAddressRead = 0;
  uint32_t mRegisterErrorGap = 0;
  int32_t mCurrentRegisterIndex = 0;
  int32_t mPreviousRegisterIndex = 0;
  std::array<int, o2::trd::constants::MAXMCMCOUNT> mMcmParsingStatus{0}; // status of what was found, errors types in the parsing
  std::array<int, 8 * 16> mcmSeen;                                       // the mcm has been seen with or with out error, local to a link
  std::array<int, 8 * 16> mcmMCM;                                        // the mcm has been seen with or with out error, local to a link
  std::array<int, 8 * 16> mcmROB;                                        // the mcm has been seen with or with out error, local to a link
  std::array<int, 8 * 16> mcmSeenMissedRegister;                         // the mcm does not have a complete set of registers, local to a link
  std::bitset<constants::MAXMCMCOUNT> mMCMHasBeenSeen;                   // the mcm has been seen with or with out error, local to a link
  std::bitset<constants::MAXHALFCHAMBER> mHCHasBeenSeen;                 // the mcm has been seen with or with out error, local to a link
  bool firsttime = false;
  InteractionRecord mIR;
  InteractionRecord mPreviousIR;
  std::time_t mConfigDate;
  std::array<int, TrapRegisters::kLastReg> mStartReg{0};      // count which register a config starts at
  std::array<int, TrapRegisters::kLastReg> mStopReg{0};       // count which register a config stops at.
  std::array<int, TrapRegisters::kLastReg> mMissedReg{0};     // count the number of missed reigsters
  std::array<int, TrapRegisters::kLastReg> mRegisterCount{0}; // register frequency, a count of how many times each register appears
  std::shared_ptr<TrapConfigEvent> mTrapConfigEvent;          // emptry trap config to store the register information.
  std::vector<MCMEvent> mMCMData;
  std::array<int32_t, constants::MAXMCMCOUNT> mMCMDataIndex;
  std::array<std::map<uint32_t, uint32_t>, TrapRegisters::kLastReg> mTrapRegistersFrequencyMap; // frequency map for values in the respective registers
  std::map<uint16_t, uint16_t> mTrapRegistersAddressIndexMap;                                   //!< map of address into mTrapRegisters, populated at the end of initialiseRegisters
  TrapConfigEventQC mQCData;
  int configcount = 0;
  ClassDefNV(TrapConfigEventParser, 1);
};

} // namespace trd
} // namespace o2
#endif
