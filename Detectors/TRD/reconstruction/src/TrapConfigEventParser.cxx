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

////////////////////////////////////////////////////////////////////////////
//                                                                        //
//  TRAP config Parser                                                    //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include <fairlogger/Logger.h>
#include "Framework/ProcessingContext.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/OutputSpec.h"
#include "Framework/DataProcessorSpec.h"

#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/TrapConfigEvent.h"
#include "DataFormatsTRD/HelperMethods.h"

#include "TRDReconstruction/TrapConfigEventParser.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <array>
#include <bitset>
#include <gsl/span>

using namespace o2::trd;

TrapConfigEventParser::TrapConfigEventParser()
{
  LOGP(info, "Creating trapconfig event object");
  mTrapConfigEvent = std::make_unique<o2::trd::TrapConfigEvent>();
  mMCMDataIndex.fill(-1);
  buildAddressMap();
}

TrapConfigEventParser::~TrapConfigEventParser()
{
}

void TrapConfigEventParser::buildAddressMap()
{
  LOGP(info, "Initing registers, build the map : ");
  for (int reg = 0; reg < TrapRegisters::kLastReg; ++reg) {
    // reindex to speed things up, this time by address, map instead of a rather large lookup table.
    auto addr = mTrapConfigEvent.get()->getRegisterAddress(reg);
    mTrapRegistersAddressIndexMap[addr] = reg;
  }
  LOGP(info, "finished buildng the map ");
}
bool TrapConfigEventParser::checkRegister(uint32_t& registeraddr, uint32_t& registerdata)
{
  mPreviousRegisterAddressRead = registeraddr;
  int32_t numberbits = 0;
  std::string regname = "";
  int32_t newregidx = 0;
  bool badregister = true;
  getRegisterByAddr(registeraddr, regname, newregidx, numberbits);
  // Are the registers sequential
  if (newregidx != mLastRegIndex + 1 && newregidx != 0) {
    if (mCurrentMCMID > constants::MAXMCMCOUNT) {
      LOGP(warning, "MCMID is : {}, out of bounds", mCurrentMCMID);
      return false;
    }
    setMCMParsingStatus(mCurrentMCMID, kTrapConfigEventRegisterGap); // missing registers
    for (int miss = mLastRegIndex; miss < newregidx; ++miss) {
      mMissedReg[miss]++; // count the gaps, we count the start and stop point of gaps elsewhere
    }
    // TODO add to QC
  }
  mLastRegIndex = newregidx;
  if (numberbits >= 0 || regname != "" || newregidx >= 0) {
    // this is a bogus or unknown register
    mCurrentRegisterIndex = newregidx;
    if (mCurrentRegisterIndex < mPreviousRegisterIndex) {
      setMCMParsingStatus(mCurrentMCMID, kTrapConfigEventNoEnd);
      mStopReg[mPreviousRegisterIndex]++;
      mStartReg[mCurrentRegisterIndex]++;
      // TODO add to QC
      for (int miss = mRegistersReadForCurrentMCM; miss < newregidx; ++miss) {
        mMissedReg[miss]++; // count the gaps, we count the start and stop point of gaps elsewhere
                            //   mcmMissedRegister[currentrob * constants::NMCMROB + currentmcm].set(miss);
      }
    } else {
      if (registeraddr != mTrapConfigEvent.get()->getRegAddrByIdx(mRegistersReadForCurrentMCM)) {
        // if (registeraddr != std::get<1>(TrapRegisterMap[mRegistersReadForCurrentMCM]))
        mRegisterErrorGap += abs((int)mRegistersReadForCurrentMCM - (int)newregidx) + 1; // +1 as reg index is zero based.
        setMCMParsingStatus(mCurrentMCMID, kTrapConfigEventNoBegin);                     // no end
        // TODO add to QC
        mRegistersReadForCurrentMCM = newregidx + 1;
        for (int miss = mRegistersReadForCurrentMCM; miss < newregidx; ++miss) {
          mMissedReg[miss]++; // count the gaps, we log the start and stop point of gaps elsewhere
        }
        // TODO add to QC
      } else {
        mRegistersReadForCurrentMCM++;
        badregister = false;
      }
    }
    mCurrentRegisterWordsCount++;
  } else {
    LOGP(debug, "bad register : name:'{}' newregindex:{} numberofbits:{}, lastregindex:{} registeraddr:{:08x} ?= ", regname, newregidx, numberbits, mPreviousRegisterIndex, registeraddr);
    // TODO add to QC
  }
  mPreviousRegisterIndex = mCurrentRegisterIndex;
  return badregister;
}

void TrapConfigEventParser::compareToTrackletsHCID(std::bitset<1080> trackletshcid)
{
  // loop over config hcid and if its not present check if the config event had tracklets.
  // this is of course not conclusive but a start.
  /*  for (int i = 0; i < constants::MAXCHAMBER; ++i) {
      if (mTrapConfigEvent->isHCIDPresent(i)) {
        if (trackletshcid.test(i)) {
          LOGP(debug, "Config event had tracklets for hcid {}", i);
        } else {
          LOGP(debug, "Config event had no tracklets for hcid {}", i);
        }
      } else {
        if (trackletshcid.test(i) && !mTrapConfigEvent->isHCIDPresent(i)) {
          LOGP(debug, "No Config event but we have tracklets for HCID  {}", i);
        }
      }
    }*/
}

void TrapConfigEventParser::printMCMRegisterCount(int hcid)
{
  int roboffset = 1;
  if (hcid % 2 == 0) {
    roboffset = 0;
  }
  std::stringstream errorMCM;
  LOGP(info, "bp rob for hcid : {}....", hcid);
  for (int robidx = roboffset; robidx < 8; robidx += 2) {
    std::stringstream display;
    display << "bp rob:" << robidx << " ";
    for (int mcmidx = 0; mcmidx < 16; ++mcmidx) {
      display << fmt::format("[{:04} ({:04})] ", mcmSeen[robidx * 16 + mcmidx], mcmSeenMissedRegister[robidx * constants::NROBC1 + mcmidx]);
      if (mcmSeen[robidx * 16 + mcmidx] != 433)
        errorMCM << fmt::format("[{:04} ({:04}) == hcid:{} mcm:{} rob:{} ] ", mcmSeen[robidx * constants::NROBC1 + mcmidx], mcmSeenMissedRegister[robidx * constants::NROBC1 + mcmidx], hcid, mcmMCM[robidx * constants::NROBC1 + mcmidx], mcmROB[robidx * constants::NROBC1 + mcmidx]);
    }
    LOG(info) << display.str();
  }
  LOG(info) << errorMCM.str();
  LOG(info) << "bp rob ....";
}

void TrapConfigEventParser::unpackBlockHeader(uint32_t& header, uint32_t& registerdata, uint32_t& step, uint32_t& bwidth, uint32_t& nwords, uint32_t& registeraddr, uint32_t& exit_flag)
{
  registerdata = (header >> 2) & 0xFFFF; // 16 bit data
  step = (header >> 1) & 0x0003;
  bwidth = ((header >> 3) & 0x001F) + 1;
  // check that the bit width matches what is should be
  nwords = (header >> 8) & 0x00FF;
  registeraddr = (header >> 16) & 0xFFFF;
  exit_flag = (step == 0) || (step == 3) || (nwords == 0);
}

bool TrapConfigEventParser::parse(std::vector<uint32_t>& data)
{
  // data comes in as ir, digithcheaderall, data payload.
  uint32_t start = 0;
  uint32_t end = 0;
  int position = 0;
  while (position < data.size()) {
    position += 2;
    DigitHCHeader digithcheader;
    digithcheader.word = data[position++];
    DigitHCHeader1 digithcheader1;
    digithcheader1.word = data[position++];
    DigitHCHeader2 digithcheader2;
    digithcheader2.word = data[position++];
    DigitHCHeader3 digithcheader3;
    digithcheader3.word = data[position++];
    mCurrentHCTime = (int)digithcheader1.ptrigcount;
    auto headerwords = digithcheader.numberHCW;
    // all 4 headers are sent, unfilled ones come in as zero.
    uint32_t length = data[position++];
    //  printDigitHCHeader(a, &data[digithcheaderextra]);
    mCurrentHCID = HelperMethods::getHCIDFromDigitHCHeader(digithcheader);
    start = position;
    end = start + length;
    //   mTrapConfigEvent->isHCIDPresent(mCurrentHCID);
    //   LOGP(debug, "HCIDHCIDP HC={}", mCurrentHCID);
    if (mHCHasBeenSeen.test(mCurrentHCID)) {
      // we have already seen this HC, so we must be on a new event.
      LOGP(debug, "HC based analysis because of hc : {}", mCurrentHCID);
      analyseEventBaseStats();
      clearEventBasedStats();
    }
    mHCHasBeenSeen.set(mCurrentHCID);
    parseLink(data, start, end);
    position += end - start;
    auto trailer1 = data[position++];
    auto trailer2 = data[position++];
    LOGP(debug, "Trailers : {:08x} {:08x}", trailer1, trailer2);
  }
  return true;
}

int TrapConfigEventParser::parseSingleData(std::vector<uint32_t>& data, uint32_t header, uint32_t& idx, uint32_t end, bool& fastforward)
{
  uint32_t registerdata = (header >> 2) & 0xFFFF;  // 16 bit data
  uint32_t registeraddr = (header >> 18) & 0x3FFF; // 14 bit address
  uint16_t data_hi = 0;
  uint32_t err = 0;
  ++idx;
  if (registeraddr != 0x1FFF) {
    if (header & 0x02) { // check if > 16 bits
      data_hi = data[idx];
      LOGP(debug, "read {:08x} for data > 16 bits", data_hi);
      ++idx;
      err += ((data_hi ^ (registerdata | 1)) & 0xFFFF) != 0;
      registerdata = (data_hi & 0xFFFF0000) | registerdata;
    }
    auto badreg = checkRegister(registeraddr, registerdata);
    if (badreg == true) {
      fastforward = true;
      return false;
    }
    if (mCurrentMCMID < o2::trd::constants::MAXMCMCOUNT && mRegistersReadForCurrentMCM - 1 < TrapRegisters::kLastReg) {
      LOGP(debug, "Adding single register {:08x} [{:08x}] name: {} for mcm {}, mCurrentRegisterWordsCount {} mRegistersReadForCurrentMCM:{} regindex {} with badreg:{}", registeraddr, registerdata, getRegNameByAddr(registeraddr), mCurrentMCMID, mCurrentRegisterWordsCount, mRegistersReadForCurrentMCM, idx, badreg);
      // update frequency map:
      if (mRegistersReadForCurrentMCM > 0) {
        setRegister(mRegistersReadForCurrentMCM, mCurrentMCMID, registerdata);
        auto regmax = mTrapConfigEvent.get()->getRegisterMax(mRegistersReadForCurrentMCM - 1);
        if (registerdata > regmax) {
          LOGP(debug, "assumed corrupted data as register data is greater than the mask : {:08x} for max {:08x} registername : {} regaddress : {}", registerdata, regmax, getRegNameByIdx(mRegistersReadForCurrentMCM - 1), registeraddr);
          // TODO put into QC
          mQCData.setStopRegister(mCurrentMCMID, mRegistersReadForCurrentMCM);
        }
      }
      mRegisterCount[getRegIndexByAddr(registeraddr)]++; // keep a count of seen and accepted registers
      if (getMCMParsingStatus(mCurrentMCMID) | (kTrapConfigEventNoEnd | kTrapConfigEventAllGood)) {
        // this handles the gaps in registers, where it might be good (1) before and after the gap, but this should stay with status of gap.
        setMCMParsingStatus(mCurrentMCMID, kTrapConfigEventAllGood);
      }
    }
    if (idx >= end && data[idx] != constants::CONFIGEVENTBLOCKENDMARKER) {
      LOGP(debug, "(single-write): no more data, missing end marker Config leaving parsing due to no more data at line {}", __LINE__);
      // TOOD put into QC.
      return false;
    }

  } else {
    LOGP(warn, "Config while parsing leaving parsing due to 1fff as addr");
    return err;
  }

  return true;
}

int TrapConfigEventParser::parseBlockData(std::vector<uint32_t>& data, uint32_t header, uint32_t& idx, uint32_t end, bool& fastforward)
{
  uint32_t step = 0, bwidth = 0, nwords = 0, err = 0, exit_flag = 0;
  uint32_t registeraddr;
  uint32_t registerdata;
  uint32_t msk = 0;
  int32_t bitcnt = 0;
  unpackBlockHeader(header, registerdata, step, bwidth, nwords, registeraddr, exit_flag);
  LOGP(debug, "unpacked, registeraddr:{:08x} step : {} bwidth {} nwords {} exit_flag {} registerdata {}", registeraddr, step, bwidth, nwords, exit_flag, registerdata);
  if (isValidAddress(registeraddr)) {
    auto trapregindex = getRegIndexByAddr(registeraddr);
    if (bwidth != mTrapConfigEvent.get()->getRegisterNBits(trapregindex)) {
      // check that the bit width matches what it should be
      //  something is corrupt. What we read does not match what we expect.
      //  log to info for now until figured out.
      LOGP(warn, " probably corrupt data : bwidth of {} does not match expected bandwidth of {} for reg {} registeraddr of : {:08x} registerindex : {}", bwidth, mTrapConfigEvent.get()->getRegisterNBits(trapregindex), mTrapConfigEvent.get()->getRegisterName(trapregindex), registeraddr, trapregindex);
      // TODO bail out but how far ? just mcm or whole link?
      // TODO put into qc
    }
  } else {
    LOGP(debug, "trapreg address {:08x} is not valid", registeraddr);
    idx++;
    // something wrong jump over this data word.
    // TODO put into qc
  }
  if (exit_flag) {
    LOGP(debug, "Exit flag found.");
    // TODO put into qc
    fastforward = true;
    return err;
  }
  LOGP(debug, "bwidth {}: read {:08x}  ", bwidth, data[idx]);
  if (bwidth == 31 || (bwidth > 4 && bwidth < 8) || bwidth == 10 || bwidth == 15) {
    // TODO the part after 31 is probably not required given the above if statement of bwidth, when its out mechanism is figured out.
    //  only possible values for blocks of registers is 5, 6, 7, 10, 15, and 31
    msk = (1 << bwidth) - 1;
    bitcnt = 0;
    while (nwords > 0) {
      if (bwidth == 31) {
        ++idx;
      }
      --nwords;
      bitcnt -= bwidth;
      err += (data[idx] & 1);
      if (bwidth != 31 && bitcnt < 0) { // handle the settings for when there are multiple registers packed into 1 word
        LOGP(debug, "block next data [{}] {:08x} at line {} nwords {}", idx, data[idx], __LINE__, nwords);
        header = data[idx];
        idx++;
        err += (header & 1);
        header = header >> 1;
        bitcnt = 31 - bwidth;
        registerdata = header & (1 << bwidth) - 1;
        LOGP(debug, "block next data registerdata : {:08x} header: {} bwidth {} at line {} nwords {}", registerdata, header, bwidth, idx, data[idx], __LINE__, nwords);
      }
      auto badreg = checkRegister(registeraddr, registerdata);
      if (badreg == true) {
        fastforward = true;
        continue;
      }
      if (mCurrentMCMID < o2::trd::constants::MAXMCMCOUNT && mRegistersReadForCurrentMCM < TrapRegisters::kLastReg) {
        LOGP(debug, "Adding block register {:09x} [{:08x}] name: {}  for mcm {} mCurrentRegisterWordsCount {} mRegistersReadForCurrentMCM {} regindex {} header {:08x} with badreg:{}", registeraddr, registerdata, getRegNameByAddr(registeraddr), mCurrentMCMID, mCurrentRegisterWordsCount, mRegistersReadForCurrentMCM, idx, header, badreg);
        if (mRegistersReadForCurrentMCM > 0) {
          setRegister(mRegistersReadForCurrentMCM, mCurrentMCMID, registerdata);
          auto regmax = mTrapConfigEvent.get()->getRegisterMax(mRegistersReadForCurrentMCM - 1);
          if (registerdata <= regmax) {
            mTrapRegistersFrequencyMap[mRegistersReadForCurrentMCM - 1][registerdata]++;
          } else {
            LOGP(warn, "{} assumed corrupted data as register data is greater than the mask : {:08x} for max {:08x} registername : {} regaddress : {}", __LINE__, registerdata, regmax, getRegNameByIdx(mRegistersReadForCurrentMCM - 1), registeraddr);
          }
        }
      } else {
        LOGP(debug, "if statement failed for currentmcmregister {:08x}  registerdata {:08x}", mRegistersReadForCurrentMCM, registerdata);
        // TODO send to qc. if (mCurrentMCMID < o2::trd::constants::MAXMCMCOUNT && mRegistersReadForCurrentMCM < TrapRegisters::kLastReg)
      }

      mRegisterCount[getRegIndexByAddr(registeraddr)]++; // keep a count of seen and accepted registers
      if (getMCMParsingStatus(mCurrentMCMID) | (kTrapConfigEventAllGood | kTrapConfigEventNoEnd)) {
        // this handles the gaps in registers, where it might be good (1) before and after the gap, but this should stay with status of gap.
        setMCMParsingStatus(mCurrentMCMID, kTrapConfigEventRegisterGap);
      }
      registeraddr += step;
      header = header >> bwidth; // this is not used for the bwidth=31 case
      if (idx >= end) {
        LOGP(error, "no end markermore data, {} words read Config parsing getting out due to end of data at line : {}", idx, __LINE__);
        return false;
      }
    }
  } else {
    // must be corrupt data header.
    LOGP(debug, "Unknown bwidth from block header of {}", bwidth);
    // TODO put into QC
    setMCMParsingStatus(mCurrentMCMID, kTrapConfigEventCorruptData);
  }
  if (data[idx] != constants::CONFIGEVENTBLOCKENDMARKER) {
    LOGP(debug, "increment idx to {} due trailer of block data data[{}]={:08x} to data[{}]={:08x}", idx + 1, idx, data[idx], idx + 1, data[idx + 1]);
    ++idx; // jump over the block trailer
  }
  if (data[idx + 1] == constants::CONFIGEVENTBLOCKENDMARKER || data[idx] == 0x00007fff) {
    LOGP(debug, "end of block ignoring bogus : {:08x}?", data[idx]);
    // TODO put into QC
  }

  return idx;
}

void TrapConfigEventParser::addMCM(const int mcmid)
{
  mMCMData.emplace_back(MCMEvent(mcmid));
  mQCData.addMCM(mcmid);
  mMCMDataIndex[mcmid] = mMCMData.size() - 1;
}

void TrapConfigEventParser::analyseEventBaseStats()
{
  // check which links have not produced data.
  // check which mcm have not produced any data.
  uint32_t mcm = 0;
  std::stringstream missingMCM;
  std::stringstream missingHC;
  std::stringstream missingHCdsl;
  uint32_t countmcm = 0;
  uint32_t counthc = 0;
  LOGP(info, "Analysing event based stats");
  missingMCM << fmt::format(" missing mcm : ");
  missingHC << fmt::format(" missing halfchambers : ");
  while (mcm < constants::MAXMCMCOUNT) {
    if (!mMCMHasBeenSeen.test(mcm)) {
      missingMCM << fmt::format(" [{}", mcm); // the start of a sequence

      mcm++;
      while (mcm < constants::MAXMCMCOUNT && !mMCMHasBeenSeen.test(mcm)) {
        // jump over a block of unseen ones
        mcm++;
      }
      countmcm++;
      missingMCM << fmt::format("-{}],", mcm - 1);
    } else {
      // we can ignore this mcm (for the purposes of display) as it was present
      mcm++;
      countmcm++;
    }
  }
  // we now have a string of missing mcm in [?-?],[?-?], format
  int hc = 0;
  while (hc < constants::MAXHALFCHAMBER) {
    if (!mHCHasBeenSeen.test(hc)) {
      missingHC << fmt::format(" [{}", hc); // the start of a sequence
      hc++;
      while (hc < constants::MAXHALFCHAMBER && !mHCHasBeenSeen.test(hc)) {
        // jump do nothing till we get to the next valid one ....
        hc++;
      }
      counthc++;
      missingHC << fmt::format(" -{}]", hc - 1); // the start of a sequence
    } else {
      // we can ignore this mcm as it was present
      hc++;
      counthc++;
    }
  }
  // we now have a string of missing mcm in [?-?],[?-?], format

  // so display them
  LOG(info) << missingHC.str();
  LOGP(info, " Total HalfChambers : {} ", counthc);
  LOG(info) << missingMCM.str();
  LOGP(info, " Total MCMs : {} ", countmcm);
}

void TrapConfigEventParser::clearEventBasedStats()
{
  // rest those stats that should not span a "complete" set of config events.
  LOGP(info, "HC Count : {}", mHCHasBeenSeen.count());
  LOGP(info, "MCM Count : {}", mMCMHasBeenSeen.count());
  mMCMHasBeenSeen.reset();
  mHCHasBeenSeen.reset();
}

int TrapConfigEventParser::parseLink(std::vector<uint32_t>& data, uint32_t start, uint32_t end)
{
  uint32_t step, bwidth, nwords, idx, err, exit_flag;
  int32_t bitcnt, werr;
  uint16_t registeraddr;
  uint32_t registerdata, msk, header, data_hi, rawdat;
  mcmSeen.fill(-1);
  mcmMCM.fill(-1);
  mcmROB.fill(-1);
  mcmSeenMissedRegister.fill(-1);
  // mcmheader, header, then data
  idx = 0; // index in the packed configuration
  err = 0;
  mLastRegIndex = 0;
  int previdx = 0;
  int datalength = data.size();
  bool dualmcm = false;
  int rawidx = 0;
  mRegisterErrorGap = 0;
  mCurrentRegisterIndex = 0;
  mPreviousRegisterIndex = 0;
  bool endmarker = false;
  bool mcmendmarker = false;
  bool hcendmarker = false;
  bool fastforward = false;
  idx = start;
  bool firstfastforward = true;
  while (idx < end) {
    LOGP(debug, "************** Raw data : data[{}] = {:08x}", idx, data[idx]);
    if (fastforward) {
      LOGP(debug, "** fastforward on data ::  Raw data : data[{}] = {:08x}", idx, data[idx]);
      // loop until the next digitmcmheader, or end.
      // search for :
      //  a : end marker
      //  b : mcmheader
      //  c : end of data
      while (data[idx] && fastforward) {
        if (firstfastforward) {
          LOGP(debug, "fastforwaring from idx:{} for mcm {}", idx, mCurrentMCMID);
          // TODO put into qc
          firstfastforward = false;
        }
        // read until we find an end marker, i gnoring the data coming in so as not to pollute the configs.
        if (data[idx] == constants::CONFIGEVENTBLOCKENDMARKER) {
          ++idx; // go past the end marker with the expectation of a DigitMCMHeader to come next.
          endmarker = true;
          fastforward = false;
          continue;
        } else {
          // LOGP(warn, " no end marker found ");
          mMcmParsingStatus[mCurrentMCMID] |= kTrapConfigEventNoEnd;
          // TODO put into QC
        }
        ++idx;
      }
      fastforward = false;
      continue;
    }
    if (idx == start || mcmendmarker) {
      // mcm header
      mCurrentMCMHeader.word = data[idx];
      LOGP(debug, "header at data[{}] had : {:06} registers, last register read : {:08x} ({}) and a register gap of {}", idx, mCurrentRegisterWordsCount, mPreviousRegisterAddressRead, getRegNameByAddr(mPreviousRegisterAddressRead), mRegisterErrorGap);
      LOGP(debug, "HC end marker ?? {:08x} {:08x} {:08x} {:08x}", data[idx], data[idx + 1], data[idx + 2], data[idx + 3]);

      if (data[idx] == constants::CONFIGEVENTBLOCKENDMARKER || data[idx] == 0xeeeeeeee) {
        LOGP(debug, " we have a the first part of a config event block end marker");
        if (data[idx + 1] == constants::CONFIGEVENTBLOCKENDMARKER || data[idx] == 0xeeeeeeee) {
          LOGP(debug, " yip we have a double config event block end marker");
        }
        while (idx < end && data[idx] == 0xeeeeeeee) {
          idx++;
        }
        break; // we are done with this link;
      }
      // printDigitMCMHeader(mCurrentMCMHeader);
      if (idx != start) {
        int index = mCurrentMCMHeader.rob * constants::NMCMROB + mCurrentMCMHeader.mcm;
        if (index < constants::NMCMROB * constants::NROBC1) {
          mcmSeen[mCurrentMCMHeader.rob * constants::NMCMROB + mCurrentMCMHeader.mcm] = mCurrentRegisterWordsCount; // registers seen for this mcm
          LOGP(debug, "mcmSeen updated at {}, with value : {} rob {} * 16 + mcm {} ", mCurrentMCMHeader.rob * constants::NMCMROB + mCurrentMCMHeader.mcm, mCurrentRegisterWordsCount, (uint32_t)mCurrentMCMHeader.rob, (uint32_t)mCurrentMCMHeader.mcm);
          mcmSeenMissedRegister[mCurrentMCMHeader.rob * constants::NMCMROB + mCurrentMCMHeader.mcm] = mRegisterErrorGap; // registers missed for this mcm
          mcmMCM[mCurrentMCMHeader.rob * constants::NMCMROB + mCurrentMCMHeader.mcm] = mCurrentMCMHeader.mcm;            // registers seen for this mcm
          mcmROB[mCurrentMCMHeader.rob * constants::NMCMROB + mCurrentMCMHeader.mcm] = mCurrentMCMHeader.rob;            // registers seen for this mcm
        }
      }
      mCurrentRegisterWordsCount = 0;                                                                                    // count of register words read for this mcm
      mRegistersReadForCurrentMCM = 0;                                                                                   // count of registers read for this mcm
      mRegisterErrorGap = 0;                                                                                             // reset the count as we are starting a fresh
      mPreviousRegisterIndex = 0;                                                                                        // register index of the last register read.
      mCurrentRegisterIndex = 0;                                                                                         // index of the current register
      mCurrentMCMID = HelperMethods::getMCMId(mCurrentHCID / 2, (int)mCurrentMCMHeader.rob, (int)mCurrentMCMHeader.mcm); // (mCurrentHCID / 2) * 128 + mCurrentMCMHeader.rob * constants::NMCMROB + mCurrentMCMHeader.mcm; // current rob is 0-7/0-5 and currentmcm is 0-16.
      LOGP(debug, "MCMMCMP MCM:{} hcid: {}   rob:{} mcm: {} idx : {} endmarker : {} eventcounter: {}", mCurrentMCMID, mCurrentHCID, (int)mCurrentMCMHeader.rob, (int)mCurrentMCMHeader.mcm, idx, endmarker, (uint32_t)mCurrentMCMHeader.eventcount);

      if (mMCMHasBeenSeen.test(mCurrentMCMID)) {
        // we are now on a new trapconfig event
        LOGP(debug, "MCM based analysis because of mcm : {}", mCurrentMCMID);
        // analyseEventBaseStats();
        dualmcm = true;
      }

      mMCMHasBeenSeen.set(mCurrentMCMID);

      // new mcm so add to 2 index and data.
      addMCM(mCurrentMCMID);
      setMCMParsingStatus(mCurrentMCMID, kTrapConfigEventAllGood);
      if (data[idx] == 0) {
        LOGP(debug, "Breaking as a zero after endmarker idx: {} ", idx);
        // TODO put into QC.
        LOGP(debug, "header at data[{}] had : {:06} registers, last register read : {:08x} ({}) and a register gap of {}", idx, mCurrentRegisterWordsCount, mPreviousRegisterAddressRead, getRegNameByAddr(mPreviousRegisterAddressRead), mRegisterErrorGap);
        break;
      }
      mcmendmarker = false;
      mLastRegIndex = 0;
      ++idx;
      continue; // dont parse and go back through the loop
    }
    if (data[idx] == 0x7fff00fe) {
      if (data[idx + 1] == 0x0) {
        LOGP(debug, "MCM End Marker [{}]  {:08x} {:08x} {:08x}", idx, data[idx], data[idx + 1], data[idx + 2]);
        // end of an mcm
        mcmendmarker = true;
        idx += 2; // jump over both tailing markers
        while (data[idx] == 0 && idx < end) {
          idx++;
        }
        continue;
      } else {
        mcmendmarker = true;
        idx++;
        LOGP(debug, "MCM End Marker but the following 0x0 is missing");
        while (data[idx] == 0 && idx < end) {
          idx++;
        }
        while (data[idx] == 0 && idx < end) {
          idx++;
        }
        continue;
      }
    }
    if (data[idx] == constants::CONFIGEVENTBLOCKENDMARKER && data[idx + 1] == constants::CONFIGEVENTBLOCKENDMARKER) {
      // end marker next value which should be a header.
      LOGP(debug, "after end marker Header :  address:{0:08x}  words: {1:08x} width: {2:08x} astep: {3:08x} zero: {4:08x} at idx : {5:0d} gap: {6:0d}", (data[idx + 1] >> 16) & 0xffff, (data[idx + 1] >> 8) & 0xff, (data[idx + 1] >> 3) & 0x1f, (data[idx + 1] >> 1) & 0x3, data[idx + 1] & 0x1, idx, idx - previdx);
      previdx = idx;
      hcendmarker = true;
      ++idx;
      if (idx + 1 == end) {
        // end of half chamber
        LOGP(debug, "end of block idx : {} mcmid: {} hcid:{}", idx, mCurrentMCMID, mCurrentHCID);
      } else {
      }
      if (data[idx] == constants::CONFIGEVENTBLOCKENDMARKER) {
        ++idx; // handle the second case of the previous if statement.
      }
      continue;
    }
    header = data[idx];
    LOGP(debug, "************** Raw data : data[{}] = {:08x} at {}", idx, data[idx], __LINE__);
    if (header & 0x01) { // single data
      /**********
      SINGLE DATA
      ***********/
      parseSingleData(data, header, idx, end, fastforward);
    } else {
      /*********
      BLOCK DATA
      **********/
      parseBlockData(data, header, idx, end, fastforward);
    } // end block case
  }   // end while
  // printMCMRegisterCount(mCurrentHCID);
  if (dualmcm) {
    analyseEventBaseStats();
    clearEventBasedStats();
    mTrapConfigEventCounter++;
  }
  return false; // we only get here if the max length of the block reached
}

bool TrapConfigEventParser::setRegister(const uint32_t regidx, const uint32_t mcmid, const uint32_t registerdata)
{
  // check MCMEvent has said mcm.
  int32_t index;
  if (mcmid > constants::MAXMCMCOUNT) {
    LOGP(warn, "MCMID is : {}", mcmid);
    return false;
  }
  if (mMCMDataIndex[mcmid] == -1) {
    // not in the index add to mcmdata and create the index :
    LOGP(warn, "index for mcm : {} is invalid in setRegister", mcmid);
    return false;
  }
  index = mMCMDataIndex[mcmid];
  TrapRegInfo reginfo = mTrapConfigEvent.get()->getRegisterInfo(regidx - 1);
  mMCMData[index].setRegister(registerdata, regidx - 1, reginfo);

  return true;
}

const uint32_t TrapConfigEventParser::getRegister(const uint32_t regidx, const uint32_t mcmid)
{
  int32_t index;
  if (mMCMDataIndex[mcmid] == -1) {
    // not in the index add to mcmdata and create the index :
    LOGP(warn, "index for mcm : {} is invalid in getRegister", mcmid);
    return false;
  }
  index = mMCMDataIndex[mcmid];
  uint32_t regdata = mMCMData[index].getRegister(regidx, mTrapConfigEvent.get()->getRegisterInfo(regidx));
  return regdata;
}

int TrapConfigEventParser::flushParsingStats()
{
  mMcmParsingStatus.fill(-1);
  return 0;
}

void TrapConfigEventParser::sendTrapConfigEvent(framework::ProcessingContext& pc)
{
  LOGP(info, "About to send message with mMCMData having size : {}", mMCMData.size());
  pc.outputs().snapshot(framework::Output{o2::header::gDataOriginTRD, "TRDCFG", 0}, mMCMData);
  pc.outputs().snapshot(framework::Output{o2::header::gDataOriginTRD, "TRDCFGQC", 0}, mQCData);
  //  pc.outputs().snapshot(framework::Output{o2::header::gDataOriginTRD, "TRDCFG", 0, framework::Lifetime::Condition}, mMCMData);
  //  pc.outputs().snapshot(framework::Output{o2::header::gDataOriginTRD, "TRDCFGQC", 0, framework::Lifetime::Condition}, mQCData);
  mMCMData.clear();
  mMCMDataIndex.fill(-1); // clear();
  mQCData.clear();
}

// this is here to be able to use the on the fly index.
const int32_t TrapConfigEventParser::getRegIndexByAddr(unsigned int addr)
{
  if (isValidAddress(addr)) {
    return mTrapRegistersAddressIndexMap[addr];
  } else
    return -1;
}

// this is here to be able to use the on the fly index.
bool TrapConfigEventParser::isValidAddress(uint32_t addr)
{
  auto search = mTrapRegistersAddressIndexMap.find(addr);
  return (search != mTrapRegistersAddressIndexMap.end());
}

// this is here to be able to use the on the fly index.
const std::string TrapConfigEventParser::getRegNameByAddr(uint16_t addr)
{
  std::string name = "";
  if (auto search = mTrapRegistersAddressIndexMap.find(addr); search != mTrapRegistersAddressIndexMap.end()) {
    name = mTrapConfigEvent.get()->getRegisterName(mTrapRegistersAddressIndexMap[addr]);
  }
  return name;
}

void TrapConfigEventParser::getRegisterByAddr(uint32_t registeraddr, std::string& regname, int32_t& newregidx, int32_t& numberbits)
{
  int idx = -1;
  idx = getRegIndexByAddr(registeraddr);
  if (idx >= 0) {
    regname = mTrapConfigEvent.get()->getRegisterName(idx);
    numberbits = mTrapConfigEvent.get()->getRegisterNBits(idx);
    newregidx = idx;
  } else {
    regname = "";
    numberbits = -1;
    newregidx = -1;
  }
}
