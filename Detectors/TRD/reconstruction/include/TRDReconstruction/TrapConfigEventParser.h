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

namespace o2::trd
{

/*struct regmap_addr {
  std::string name;
  int index;
  int bitwidth;
};

struct regmap {
  std::string name;
  int address;
  int bitwidth;
};
*/

enum TrapConfigEventErrorTypes {
  kTrapConfigEventAllGood,
  kTrapConfigEventNoEnd,
  kTrapConfigEventNoBegin,
  kTrapConfigEventNoGarbageEnd,
  kTrapConfigEventNoGarbageBegin,
  kTrapConfigEventLastError
}; // possible errors for parsing.

class TrapConfigEventParser
{
  // class used to unpack the trap config events, and figure out what to do to them.
 public:
  TrapConfigEventParser();
  ~TrapConfigEventParser();

  // int getTrapReg(TrapReg reg, int det = -1, int rob = -1, int mcm = -1);

  unsigned int getDmemUnsigned(int addr, int det, int rob, int mcm);

  // helper methods
  std::string getConfigVersion() { return mTrapConfigEventVersion; }
  std::string getConfigName() { return mTrapConfigEventName; }
  void setConfigVersion(std::string version) { mTrapConfigEventVersion = version; } // these must some how be gotten from git or wingdb.
  void setConfigName(std::string name) { mTrapConfigEventName = name; }             // these must be gotten from git or wingdb.

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
  std::array<int, TrapConfigEvent::kLastReg>& getStartRegArray() { return mStartReg; }      // the number of time this register was read as the first register
  std::array<int, TrapConfigEvent::kLastReg>& getStopRegArray() { return mStopReg; }        // the number of time this register was read as the last register
  std::array<int, TrapConfigEvent::kLastReg>& getMissedRegArray() { return mMissedReg; }    // the number of times this register was not read
  std::array<int, TrapConfigEvent::kLastReg>& getRegisterCount() { return mRegisterCount; } // total count for each register
  void init(){};
  bool isNewConfig();
  TrapConfigEvent getNewConfig() { return *mTrapConfigEvent.get(); };
  TrapConfigEvent* getNewConfigPtr() { return mTrapConfigEvent.get(); };
  // TrapConfigEvent getNewConfig() { return *mTrapConfigEvent.get(); };
  // TrapConfigEvent* getNewConfigPtr() { return mTrapConfigEvent.get(); };
  void sendTrapConfigEvent(framework::ProcessingContext& pc);
  // write the current config to a file
  void writeFile(int eventcount);
  uint32_t countHCIDPresent() { return mTrapConfigEvent->countHCIDPresent(); }
  uint32_t countMCMPresent() { return mTrapConfigEvent->countMCMPresent(); }

  // flush the stats that are config event specific
  // a single parse is almost certainly not going to contain the complete event, this is then run after we think we have the complete set.
  int flushParsingStats();
  int analyseMaps();
  int analyseMcmSeen();
  int64_t getConfigSize() { return sizeof(*mTrapConfigEvent); };
  void setMCMParsingStatus(uint32_t mcmid, int status){ mMcmParsingStatus[mcmid]=status;}                     // no end
  int getMCMParsingStatus(uint32_t mcmid){ return mMcmParsingStatus[mcmid];}                     // no end

 private:
  uint32_t mCurrentHCID = 0;
  DigitMCMHeader mCurrentMCMHeader;
  uint32_t mCurrentMCMID = 0;
  uint32_t mCurrentDataIndex = 0;
  uint32_t mLastRegIndex = 0;
  uint32_t mRegistersReadForCurrentMCM = 0;
  uint32_t mCurrentRegisterWordsCount = 0;
  uint32_t mPreviousRegisterAddressRead = 0;
  uint32_t mRegisterErrorGap = 0;
  int32_t mCurrentRegisterIndex = 0;
  int32_t mPreviousRegisterIndex = 0;
  uint32_t mRegisterBase = 0;
  uint32_t mOffsetToRegister = 0;
  //  std::map<int, std::tuple<std::string, int, int>> TrapRegisterMap_addr;
  // std::array<int, 0xe000> mTrapRegistersAddressIndex; // index by address into mTrapRegisters.
  std::array<int, o2::trd::constants::MAXMCMCOUNT> mMcmParsingStatus{0};                                  // status of what was found, errors types in the parsing
                                                                                                          //  std::array<uint32_t, o2::trd::constants::MAXHALFCHAMBER> mHalfChamberLastSeen;                          // timestamp
                                                                                                          //  std::array<uint32_t, o2::trd::constants::MAXHALFCHAMBER> mHalfChamberSeenSinceLastWritten;              // timestamp
                                                                                                          //  std::array<uint32_t, o2::trd::constants::MAXHALFCHAMBER> mHalfChamberFrequencyInAccumulation;           // frequency in accumulation period.
                                                                                                          //  std::array<uint32_t, o2::trd::constants::MAXMCMCOUNT> mMCMLastSeen;                                     // timestamp
                                                                                                          //  std::array<uint32_t, o2::trd::constants::MAXMCMCOUNT> mMCMSeenSinceLastWritten;                         // timestamp
                                                                                                          //  std::array<uint32_t, o2::trd::constants::MAXMCMCOUNT> mMCMFrequencyInAccumulation;                      // frequency in accumulation.
  std::array<int, 8 * 16> mcmSeen;                                                                        // the mcm has been seen with or with out error, local to a link
  std::array<int, 8 * 16> mcmMCM;                                                                         // the mcm has been seen with or with out error, local to a link
  std::array<int, 8 * 16> mcmROB;                                                                         // the mcm has been seen with or with out error, local to a link
  std::array<int, 8 * 16> mcmSeenMissedRegister;                                                          // the mcm does not have a complete set of registers, local to a link
  bool firsttime = false;
  //  static bool mRegisterAddressMapInitialised;
  InteractionRecord mIR;
  InteractionRecord mPreviousIR;
  std::time_t mConfigDate;
  std::array<int, TrapConfigEvent::kLastReg> mStartReg{0};      // count which register a config starts at
  std::array<int, TrapConfigEvent::kLastReg> mStopReg{0};       // count which register a config stops at.
  std::array<int, TrapConfigEvent::kLastReg> mMissedReg{0};     // count the number of missed reigsters
  std::array<int, TrapConfigEvent::kLastReg> mRegisterCount{0}; // register frequency, a count of how many times each register appears
  std::string mTrapConfigEventName;                             // TOOD figure how to pull this in or seperately put it in the CCDB
  std::string mTrapConfigEventVersion;
  // TrapConfigEvent mTrapConfigEvent;
  // TrapConfigEvent mCCDBTrapConfigEvent;
  //std::shared_ptr<TrapConfigEventMessage> mTrapConfigEventMessage;
  std::shared_ptr<TrapConfigEvent> mTrapConfigEvent;
  // std::shared_ptr<TrapConfigEvent> mCCDBTrapConfigEvent;
  std::array<std::map<uint32_t, uint32_t>, TrapConfigEvent::kLastReg> mTrapRegistersFrequencyMap; // frequency map for values in the respective registers
  std::map<uint32_t, std::map<uint32_t, uint32_t>> mTrapValueFrequencyMap;                        // count of different value in the registers for a mcm,register used to find most frequent value.   Not needed here, as this is now 1 time frame, it will be used in the aggregator.
  int configcount = 0;
  ClassDefNV(TrapConfigEventParser, 1);
};

} // namespace o2::trd
#endif
