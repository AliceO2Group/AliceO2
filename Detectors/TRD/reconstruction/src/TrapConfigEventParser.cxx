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
#include "DataFormatsTRD/HelperMethods.h"

#include "TRDReconstruction/TrapConfigEventParser.h"

#include <TH2F.h>
#include <TFile.h>
#include <TCanvas.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <array>
#include <bitset>

using namespace o2::trd;

TrapConfigEventParser::TrapConfigEventParser()
{
  LOGP(info, "Creating trapconfig event object");
  mTrapConfigEvent = std::make_shared<TrapConfigEvent>();
}

TrapConfigEventParser::~TrapConfigEventParser()
{
}

bool TrapConfigEventParser::checkRegister(uint32_t& registeraddr, uint32_t& registerdata)
{
  mPreviousRegisterAddressRead = registeraddr;
  int32_t numberbits = 0;
  std::string regname = "";
  int32_t newregidx = 0;
  bool badregister = true;
  mTrapConfigEvent.get()->getRegisterByAddr(registeraddr, regname, newregidx, numberbits);
  mOffsetToRegister = mRegisterBase + newregidx;
  // Are the registers sequential
  if (newregidx != mLastRegIndex + 1 && newregidx != 0) {
    mTrapConfigEventMessage->setMCMParsingStatus(mCurrentMCMID, 5); // missing registers
    for (int miss = mLastRegIndex; miss < newregidx; ++miss) {
      mMissedReg[miss]++; // count the gaps, we count the start and stop point of gaps elsewhere
    }
    LOGP(warn, " register step mismatch : went from {:06x} ({}) i:{} to {:06x} ({}) i:{} and mcmid: {} mMcmParsingStatus[{}]=5", mPreviousRegisterAddressRead, mTrapConfigEvent.get()->getRegNameByAddr(mPreviousRegisterAddressRead), mLastRegIndex, registeraddr, mTrapConfigEvent.get()->getRegNameByAddr(registeraddr), newregidx, registerdata, mCurrentMCMID, mTrapConfigEventMessage->getMCMParsingStatus(mCurrentMCMID), mCurrentMCMID);
  }
  mLastRegIndex = newregidx;
  if (numberbits >= 0 || regname != "" || newregidx >= 0) {
    // this is a bogus or unknown register
    LOGP(debug, "good register : name:{} newregindex:{} numberofbits:{}, lastregindex:{} registeraddr:{:08x} ", regname, newregidx, numberbits, mPreviousRegisterIndex, registeraddr);
    mCurrentRegisterIndex = newregidx;
    if (mCurrentRegisterIndex < mPreviousRegisterIndex) {
      mTrapConfigEventMessage->setMCMParsingStatus(mCurrentMCMID, 4); // no end
      mTrapConfigEventMessage->setMCMParsingStatus(mCurrentMCMID, 6); // no end
      mStopReg[mPreviousRegisterIndex]++;
      mStopReg[mCurrentRegisterIndex]++;
      LOGP(warn, " current register index is less than previous, we have looped back mcmId[{}] = {} and mcmId[{}]= {} current register index {} previousregisteraddressread {} ", mCurrentMCMID, mTrapConfigEventMessage->getMCMParsingStatus(mCurrentMCMID), mCurrentMCMID + 1, mTrapConfigEventMessage->getMCMParsingStatus(mCurrentMCMID + 1), mCurrentRegisterIndex, mPreviousRegisterAddressRead);
      for (int miss = mRegistersReadForCurrentMCM; miss < newregidx; ++miss) {
        mMissedReg[miss]++; // count the gaps, we count the start and stop point of gaps elsewhere
                            //   mcmMissedRegister[currentrob * constants::NMCMROB + currentmcm].set(miss);
      }
    } else {
      if (registeraddr != mTrapConfigEvent.get()->getRegAddrByIdx(mRegistersReadForCurrentMCM)) {
        // if (registeraddr != std::get<1>(TrapRegisterMap[mRegistersReadForCurrentMCM]))
        mRegisterErrorGap += abs((int)mRegistersReadForCurrentMCM - (int)newregidx) + 1; // +1 as reg index is zero based.
        mTrapConfigEventMessage->setMCMParsingStatus(mCurrentMCMID, 3);                     // no end
        LOGP(debug, " get mTrapConfigEvent.get()->getRegNameByAddr( {:08x} ) at func:{} line:{}", registeraddr, __func__, __LINE__);
        auto tmpnamebyaddr = mTrapConfigEvent.get()->getRegNameByAddr(registeraddr);
        LOGP(debug, " got mTrapConfigEvent.get()->getRegNameByAddr( {:08x} ) at func:{} line:{}", registeraddr, __func__, __LINE__);
        std::string tmpnamebyidx = ""; // mTrapConfigEventMessage.get()->getRegNameByIdx(mRegistersReadForCurrentMCM);
        auto tmpregidxbyaddr = mTrapConfigEvent.get()->getRegIndexByAddr(registeraddr);
        LOGP(warn, " current register is not sequential to previous register, we have a gap : {} from {} to {} mRegisterErrorGap now : {} name compare {} ?= {} at {} mMcmParsingStatus[{}]={}", abs((int)mRegistersReadForCurrentMCM - (int)newregidx),
             mRegistersReadForCurrentMCM, tmpregidxbyaddr, mRegisterErrorGap, tmpnamebyaddr, tmpnamebyidx, __LINE__, mCurrentMCMID, mTrapConfigEventMessage->getMCMParsingStatus(mCurrentMCMID));
        mRegistersReadForCurrentMCM = newregidx + 1;
        for (int miss = mRegistersReadForCurrentMCM; miss < newregidx; ++miss) {
          mMissedReg[miss]++; // count the gaps, we count the start and stop point of gaps elsewhere
        }
        LOGP(warn, " Registers are non-sequential : {} from {} to {} mRegisterErrorGap now : {} name compare {} ?= {} at {}", abs(mRegistersReadForCurrentMCM - mTrapConfigEvent.get()->getRegIndexByAddr(registeraddr)), mRegistersReadForCurrentMCM, mTrapConfigEvent.get()->getRegIndexByAddr(registeraddr), mRegisterErrorGap, mTrapConfigEvent.get()->getRegNameByAddr(registeraddr), "" /*mTrapConfigEventMessage.get()->getRegNameByIdx(mRegistersReadForCurrentMCM)*/, __LINE__);
      } else {
        LOGP(debug, " Registers are sequential : {} from {} to {} mRegisterErrorGap now : {} name compare {} ?= {} at {}", abs(mRegistersReadForCurrentMCM - mTrapConfigEvent.get()->getRegIndexByAddr(registeraddr)), mRegistersReadForCurrentMCM, mTrapConfigEvent.get()->getRegIndexByAddr(registeraddr), mRegisterErrorGap, mTrapConfigEvent.get()->getRegNameByAddr(registeraddr), "" /*mTrapConfigEventMessage.get()->getRegNameByIdx(mRegistersReadForCurrentMCM)*/, __LINE__);
        mRegistersReadForCurrentMCM++;
        badregister = false;
      }
    }
    mCurrentRegisterWordsCount++;
  } else {
    LOGP(warn, "bad register : name:'{}' newregindex:{} numberofbits:{}, lastregindex:{} registeraddr:{:08x} ?= ", regname, newregidx, numberbits, mPreviousRegisterIndex, registeraddr);
  }
  mPreviousRegisterIndex = mCurrentRegisterIndex;
  return badregister;
}

void TrapConfigEventParser::compareToTrackletsHCID(std::bitset<1080> trackletshcid)
{
  // loop over config hcid and if its not present check if the config event had tracklets.
  // this is of course not conclusive but a start.
  for (int i = 0; i < constants::MAXCHAMBER; ++i) {
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
  }
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
    LOGP(debug, " bc : {} orbit: {}", data[position], data[position + 1]);
    position += 2;
    DigitHCHeader a;
    a.word = data[position++];
    LOGP(debug, " DigitHCHeader 0 {:08x}", a.word);
    int digithcheaderextra = position;
    LOGP(debug, " DigitHCHeader 1 {:08x}", data[position]);
    position++;
    LOGP(debug, " DigitHCHeader 2 {:08x}", data[position]);
    position++;
    LOGP(debug, " DigitHCHeader 3 {:08x}", data[position]);
    position++;
    uint32_t length = data[position++];
    LOGP(debug, " Link length : {}", length);
    // printDigitHCHeader(a, &data[digithcheaderextra]);
    mCurrentHCID = HelperMethods::getHCIDFromDigitHCHeader(a);
    start = position;
    end = start + length;
    mTrapConfigEvent->isHCIDPresent(mCurrentHCID);
    parseLink(data, start, end);
    position += end - start;
    auto trailer1 = data[position++];
    auto trailer2 = data[position++];
    LOGP(debug, "Trailers : {:08x} {:08x}", trailer1, trailer2);
    // TOOD do something with else remove:
    // std::array<int, 8 * 16> mcmSeen;                                                                   // the mcm has been seen with or with out error, local to a link
    // std::array<int, 8 * 16> mcmMCM;                                                                    // the mcm has been seen with or with out error, local to a link
    // std::array<int, 8 * 16> mcmROB;                                                                    // the mcm has been seen with or with out error, local to a link
    // std::array<int, 8 * 16> mcmSeenMissedRegister;                                                     // the mcm does not have a complete set of registers, local to a link
    // std::array<std::bitset<TrapConfigEvent::kLastReg>, 8 * 16> mcmMissedRegister;                           // bitpattern of which registers were seen and not seen for a given mcm.
  }
  // are we complete enough to now compare ?
  analyseMaps();
  analyseMcmSeen();
  return true;
}

int TrapConfigEventParser::parseSingleData(std::vector<uint32_t>& data, uint32_t header, uint32_t& idx, uint32_t end, bool& fastforward)
{
  uint32_t registerdata = (header >> 2) & 0xFFFF;  // 16 bit data
  uint32_t registeraddr = (header >> 18) & 0x3FFF; // 14 bit address
  uint16_t data_hi = 0;
  uint32_t err = 0;
  LOGP(debug, "single data raw:{:08x} idx: addr : {:08x} data : {:08x}", data[idx], idx, registeraddr, registerdata);
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
    if (mCurrentMCMID < o2::trd::constants::MAXMCMCOUNT && mRegistersReadForCurrentMCM - 1 < TrapConfigEvent::kLastReg) {
      LOGP(debug, "Adding single register {:08x} [{:08x}] name: {} for mcm {}, mCurrentRegisterWordsCount {} mRegistersReadForCurrentMCM:{} regindex {} with badreg:{}", registeraddr, registerdata, mTrapConfigEvent.get()->getRegNameByAddr(registeraddr), mCurrentMCMID, mCurrentRegisterWordsCount, mRegistersReadForCurrentMCM, idx, badreg);
      // update frequency map:
      if (mRegistersReadForCurrentMCM > 0) {
        mTrapConfigEvent.get()->setRegisterValueByIdx(registerdata, mRegistersReadForCurrentMCM - 1, mCurrentMCMID);
        // mMCMCurrentEvent[mCurrentMCMID]++; // registers seen for this mcm
        auto regdata = registerdata; // mTrapConfigEventMessage.get()->getRegisterValueByIdx(mRegistersReadForCurrentMCM-1,mCurrentMCMID);
        auto regmax = mTrapConfigEvent.get()->getRegisterMax(mRegistersReadForCurrentMCM - 1);
        if (regdata > regmax) {
          LOGP(warn, "assumed corrupted data as register data is greater than the mask : {:08x} for max {:08x} registername : {} regaddress : {}", regdata, regmax, mTrapConfigEvent.get()->getRegNameByIdx(mRegistersReadForCurrentMCM - 1), registeraddr);
        } else {
          mTrapRegistersFrequencyMap[mRegistersReadForCurrentMCM - 1][regdata]++;
          //TODO move to calibrator mTrapValueFrequencyMap[mCurrentMCMID * TrapConfigEvent::kLastReg + mRegistersReadForCurrentMCM - 1].insert(std::make_pair(regdata, 1)); // count of different value in the registers for a mcm,register used to find probably value.
        }
        mCurrentMCMRegisters[mCurrentMCMID * TrapConfigEvent::kLastReg + mRegistersReadForCurrentMCM - 1] = regdata;
        // TODO this is not saving it to the CCDBConfig at all !
        //   if(mTrapConfigEvent.get()->getRegNameByIdx(mRegistersReadForCurrentMCM-1) == "ADCMSK"){
        //   LOGP(debug,"** just added {:08x} data to _ADCMSK for mCurrentMCMID {}",registerdata,mCurrentMCMID);
        //   }
      }
      mRegisterCount[mTrapConfigEvent.get()->getRegIndexByAddr(registeraddr)]++; // keep a count of seen and accepted registers
      if (mTrapConfigEventMessage->getMCMParsingStatus(mCurrentMCMID) < 2) {
        // this handles the gaps in registers, where it might be good (1) before and after the gap, but this should stay with status of gap.
        mTrapConfigEventMessage->setMCMParsingStatus(mCurrentMCMID, 1);
      }
    }
    if (idx >= end && data[idx] != constants::CONFIGEVNETBLOCKENDMARKER) {
      LOGP(error, "(single-write): no more data, missing end marker Config leaving parsing due to no more data at line {}", __LINE__);
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
  if (mTrapConfigEvent.get()->isValidAddress(registeraddr)) {
    auto trapregindex = mTrapConfigEvent.get()->getRegIndexByAddr(registeraddr);
    if (bwidth != mTrapConfigEvent.get()->getRegisterNBits(trapregindex)) {
      // check that the bit width matches what it should be
      //  something is corrupt. What we read does not match what we expect.
      //  log to info for now until figured out. TODO take out info
      LOGP(warn, " probably corrupt data : bwidth of {} does not match expected bandwidth of {} for reg {} registeraddr of : {:08x} registerindex : {}", bwidth, mTrapConfigEvent.get()->getRegisterNBits(trapregindex), mTrapConfigEvent.get()->getRegisterName(trapregindex), registeraddr, trapregindex);
      // TODO bail out but how far ? just mcm or whole link?
    }
  } else {
    LOGP(warn, "trapreg address {:08x} is not valid", registeraddr);
  }
  if (exit_flag) {
    LOGP(warn, "Exit flag found.");
    fastforward = true;
    return err;
  }
  if (bwidth == 31 || (bwidth > 4 && bwidth < 8) || bwidth == 10 || bwidth == 15) {
    // TODO the part after 31 is probably not required given the above if statement of bwidth, when its out mechanism is figured out.
    //  only possible values for blocks of registers is 5, 6, 7, 10, 15, and 31
    msk = (1 << bwidth) - 1;
    bitcnt = 0;
    while (nwords > 0) {
      LOGP(debug, "bwidth {}: read {:08x}  ", bwidth, data[idx]);
      if (bwidth == 31)
        ++idx;
      --nwords;
      bitcnt -= bwidth;
      err += (data[idx] & 1);
      LOGP(debug, "bitcnt {}: err: {:08x}  ", bitcnt, err);
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
      if (mCurrentMCMID < o2::trd::constants::MAXMCMCOUNT && mRegistersReadForCurrentMCM < TrapConfigEvent::kLastReg) {
        LOGP(debug, "Adding block register {:09x} [{:08x}] name: {}  for mcm {} mCurrentRegisterWordsCount {} mRegistersReadForCurrentMCM {} regindex {} header {:08x} with badreg:{}", registeraddr, registerdata, mTrapConfigEvent.get()->getRegNameByAddr(registeraddr), mCurrentMCMID, mCurrentRegisterWordsCount, mRegistersReadForCurrentMCM, idx, header, badreg);
        if (mCurrentMCMRegisters[mCurrentMCMID * TrapConfigEvent::kLastReg + mRegistersReadForCurrentMCM] != registerdata) {
          if (mRegistersReadForCurrentMCM > 0) {
            mTrapConfigEvent.get()->setRegisterValueByIdx(registerdata, mRegistersReadForCurrentMCM - 1, mCurrentMCMID);
            // mMCMCurrentEvent[mCurrentMCMID]++; // registers seen for this mcm
            auto regdata = registerdata; // mTrapConfigEventMessage.get()->getRegisterValueByIdx(mRegistersReadForCurrentMCM-1,mcmid);
            auto regmax = mTrapConfigEvent.get()->getRegisterMax(mRegistersReadForCurrentMCM - 1);
            if (regdata > regmax) {
              LOGP(warn, "{} assumed corrupted data as register data is greater than the mask : {:08x} for max {:08x} registername : {} regaddress : {}", __LINE__, regdata, regmax, mTrapConfigEvent.get()->getRegNameByIdx(mRegistersReadForCurrentMCM - 1), registeraddr);
            } else {
              mTrapRegistersFrequencyMap[mRegistersReadForCurrentMCM - 1][regdata]++;
              mTrapValueFrequencyMap[mCurrentMCMID * TrapConfigEvent::kLastReg + mRegistersReadForCurrentMCM - 1][regdata]++;                         // count of different value in the registers for a mcm,register used to find probably value.
              mTrapValueFrequencyMap[mCurrentMCMID * TrapConfigEvent::kLastReg + mRegistersReadForCurrentMCM - 1].insert(std::make_pair(regdata, 1)); // count of different value in the registers for a mcm,register used to find probably value.
              mCurrentMCMRegisters[mCurrentMCMID * TrapConfigEvent::kLastReg + mRegistersReadForCurrentMCM - 1] = registerdata;
            }
            mTrapConfigEventMessage->setRegisterSeen(mCurrentMCMID, 1);
          }
        } else {
          LOGP(debug, "if statement failed for currentmcmregister {:08x}  registerdata {:08x}", mCurrentMCMRegisters[mCurrentMCMID * TrapConfigEvent::kLastReg + mRegistersReadForCurrentMCM], registerdata);
        }

        mRegisterCount[mTrapConfigEvent.get()->getRegIndexByAddr(registeraddr)]++; // keep a count of seen and accepted registers
        // if( mTrapRegisters[].CanIgnore()==false){
        //   mMcMDataHasChanged[mRegistersReadForCurrentMCM]=true;
        // }
        if (mTrapConfigEventMessage->getMCMParsingStatus(mCurrentMCMID) < 2) {
          // this handles the gaps in registers, where it might be good (1) before and after the gap, but this should stay with status of gap.
          mTrapConfigEventMessage->setMCMParsingStatus(mCurrentMCMID, 1);
        }
      } else {
        LOGP(warn, "if (mCurrentMCMRegisters[mCurrentMCMID * kLastReg + mRegistersReadForCurrentMCM] != registerdata mCurrentMCMID:{} kLastReg:{} mRegistersReadForCurrentMCM:{} mCurrentMCMRegisters[mCurrentMCMID*kLastReg+mRegistersReadForCurrentMCM]=={} != {}", mCurrentMCMID, TrapConfigEvent::kLastReg, mRegistersReadForCurrentMCM, mCurrentMCMRegisters[mCurrentMCMID * TrapConfigEvent::kLastReg + mRegistersReadForCurrentMCM], registerdata);
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
    LOGP(warn, "Unknown bwidth from block header of {}", bwidth);
  }
  if (data[idx] != constants::CONFIGEVNETBLOCKENDMARKER) {
    LOGP(debug, "increment idx to {} due trailer of block data data[{}]={:08x} to data[{}]={:08x}", idx + 1, idx, data[idx], idx + 1, data[idx + 1]);
    ++idx; // jump over the block trailer
  }
  if (data[idx + 1] == constants::CONFIGEVNETBLOCKENDMARKER || data[idx] == 0x00007fff) {
    LOGP(debug, "end of block ignoring bogus : {:08x}?", data[idx]);
  }

  return idx;
}

int TrapConfigEventParser::parseLink(std::vector<uint32_t>& data, uint32_t start, uint32_t end)
{
  mTrapConfigEvent->HCIDIsPresent(mCurrentHCID);
  uint32_t step, bwidth, nwords, idx, err, exit_flag;
  int32_t bitcnt, werr;
  uint16_t registeraddr;
  uint32_t registerdata, msk, header, data_hi, rawdat;
  mOffsetToRegister = 0; // offset of a register into the raw storage area
  mcmSeen.fill(-1);
  mcmMCM.fill(-1);
  mcmROB.fill(-1);
  mcmSeenMissedRegister.fill(-1);
  //  for(auto regbit : mcmMissedRegister){ // bitpattern of which registers were seen and not seen for a given mcm.
  //    regbit.reset(); // set bitset to 0
  //  }
  // mcmheader, header, then data
  idx = 0; // index in the packed configuration
  err = 0;
  mLastRegIndex = 0;
  int previdx = 0;
  int datalength = data.size();

  int rawidx = 0;
  mRegisterErrorGap = 0;
  mCurrentRegisterIndex = 0;
  mPreviousRegisterIndex = 0;
  /*uint32_t mCurrentHCID=0;
  DigitMCMHeader mCurrentMCMHeader;
  uint32_t mCurrentMCMID=0;
  uint32_t mCurrentDataIndex=0;
  uint32_t mLastRegIndex=0;
  uint32_t mRegistersReadForCurrentMCM = 0;
  uint32_t mCurrentRegisterWordsCount = 0;
  uint32_t mPreviousRegisterAddressRead = 0;
  uint32_t mRegisterErrorGap = 0;*/
  bool endmarker = false;
  bool fastforward = false;
  // for (idx=start;idx<end/2;++idx) {
  //   LOGP(/*info*/info, " data[{} = {:08x}]", idx, data[idx]);
  // }
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
          LOGP(warn, "fastforwaring from idx:{} for mcm {}", idx, mCurrentMCMID);
          firstfastforward = false;
        }
        // read until we find an end marker, ignoring the data coming in so as not to pollute the configs.
        if (data[idx] == constants::CONFIGEVNETBLOCKENDMARKER) {
          // LOGP(warn, " end marker found ");
          ++idx; // go past the end marker with the expectation of a DigitMCMHeader to come next.
          endmarker = true;
          fastforward = false;
          continue;
        } else {
          // LOGP(warn, " no end marker found ");
        }
        ++idx;
      }
      fastforward = false;
      continue;
    }
    if (idx == start || endmarker) {
      // mcm header
      mCurrentMCMHeader.word = data[idx];
      LOGP(debug, "header at data[{}] had : {:06} registers, last register read : {:08x} ({}) and a register gap of {}", idx, mCurrentRegisterWordsCount, mPreviousRegisterAddressRead, mTrapConfigEvent.get()->getRegNameByAddr(mPreviousRegisterAddressRead), mRegisterErrorGap);
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
      mCurrentRegisterWordsCount = 0;                                                                                // count of register words read for this mcm
      mRegistersReadForCurrentMCM = 0;                                                                               // count of registers read for this mcm
      mRegisterErrorGap = 0;                                                                                         // reset the count as we are starting a fresh
      mPreviousRegisterIndex = 0;                                                                                    // register index of the last register read.
      mCurrentRegisterIndex = 0;                                                                                     // index of the current register
      mCurrentMCMID = (mCurrentHCID / 2) * 128 + mCurrentMCMHeader.rob * constants::NMCMROB + mCurrentMCMHeader.mcm; // current rob is 0-7/0-5 and currentmcm is 0-16.
      mTrapConfigEventMessage->setMCMParsingStatus(mCurrentMCMID, 1);
      if (data[idx] == 0) {
        LOGP(warn, "Breaking as a zero after endmarker idx: {} ", idx);
        LOGP(debug, "header at data[{}] had : {:06} registers, last register read : {:08x} ({}) and a register gap of {}", idx, mCurrentRegisterWordsCount, mPreviousRegisterAddressRead, mTrapConfigEvent.get()->getRegNameByAddr(mPreviousRegisterAddressRead), mRegisterErrorGap);
        break;
      }
      // mcmSeen[mCurrentMCMHeader.rob * 16 + mCurrentMCMHeader.mcm] = 0;
      endmarker = false;
      mLastRegIndex = 0;
      mRegisterBase = mCurrentMCMID * 432;
      ++idx;
      continue; // dont parse and go back through the loop
    }
    if (data[idx] == constants::CONFIGEVNETBLOCKENDMARKER || (data[idx] == 0x7ffff && data[idx + 1] == constants::CONFIGEVNETBLOCKENDMARKER)) {
      // end marker next value which should be a header.
      LOGP(debug, "after end marker Header :  address:{0:08x}  words: {1:08x} width: {2:08x} astep: {3:08x} zero: {4:08x} at idx : {5:0d} gap: {6:0d}", (data[idx + 1] >> 16) & 0xffff, (data[idx + 1] >> 8) & 0xff, (data[idx + 1] >> 3) & 0x1f, (data[idx + 1] >> 1) & 0x3, data[idx + 1] & 0x1, idx, idx - previdx);
      previdx = idx;
      endmarker = true;
      ++idx;
      if (data[idx] == constants::CONFIGEVNETBLOCKENDMARKER) {
        ++idx; // handle the second case of the previous if statement.
      }
      continue;
    }
    header = data[idx];
    LOGP(debug, "************** Raw data : data[{}] = {:08x}", idx, data[idx]);
    /*****************
    SINGLE DATA
    *****************/
    if (header & 0x01) { // single data
      parseSingleData(data, header, idx, end, fastforward);
    } else {
      /*****************
    BLOCK DATA
    *****************/
      parseBlockData(data, header, idx, end, fastforward);
    } // end block case
  }   // end while
  // loop over which mcms never sent data.
  // printMCMRegisterCount(mCurrentHCID);
  return false; // we only get here if the max length of the block reached!
}

int TrapConfigEventParser::flushParsingStats()
{
  mTrapConfigEventMessage->clearParsingStatus();
  return 1;
}

int TrapConfigEventParser::analyseMaps()
{
  // loop through the maps of registers and determine, per mcm, per rob, per halfchamber, per chamber, constants.
  //
  // std::array<std::map<uint32_t, uint32_t>, TrapConfigEvent::kLastReg> mTrapRegistersFrequencyMap;
  if (configcount > 8) {
    uint oldmcm = 0;
    for (auto& valuemap : mTrapValueFrequencyMap) {
      LOGP(debug, "ZZZ1 valuemap.second.size():{} ", valuemap.second.size());

      for (const auto& elem : valuemap.second) {
        uint64_t mcmidreg = valuemap.first;
        uint64_t mcmid = mcmidreg / TrapConfigEvent::kLastReg;
        uint64_t regid = mcmidreg - mcmid * TrapConfigEvent::kLastReg;
        LOGP(debug, "ZZZ mcm:{} reg:{} mcmidreg:{} mcmid*lastreg:{} value:{} count:{}", mcmid, regid, mcmidreg, mcmid * TrapConfigEvent::kLastReg, elem.first, elem.second);
      }
    }
  }
  configcount++;
  return 1;
}

int TrapConfigEventParser::analyseMcmSeen()
{
  // loop through the maps of registers and determine, per mcm, per rob, per halfchamber, per chamber, constants.
  //
  //
  // std::array<uint64_t,32> mcmpresentcount{0};
  /* std::array<std::map<uint32_t, uint32_t>, TrapConfigEvent::kLastReg> mMcmMaps;
  for(auto& mcmcount : mMCMCurrentEvent){
    LOGP(info,"###mcmcount of {} ", mcmcount);
    auto idx=mcmcount;
    mMcmMaps[idx][mcmcount]++;
  }
  for (auto& mcmmap : mMcmMaps) {
    for (const auto& elem : mcmmap) {
      LOGP(info, "[{:08x}] = {}", elem.first, elem.second);
    }
  }
  int regcount=0;
  for (auto& regmap : mTrapRegistersFrequencyMap) {
    for (const auto& elem : regmap) {
      LOGP(debug, "[{:08x}] = {}", elem.first, elem.second);
    }
    regcount++;
  }
*/
  return 1;
}

bool TrapConfigEventParser::isNewConfig()
{
  //  if(mCCDBTrapConfigEvent == mTrapConfigEventMessage) return true;
  return false;
}

void TrapConfigEventParser::sendTrapConfigEvent(framework::ProcessingContext& pc)
{
  LOGP(info, "About to send message with first value of copied trapconfigevent A is tpl00 : {}", mTrapConfigEvent.get()->getRegisterValue(0, 31361));
  pc.outputs().snapshot(framework::Output{o2::header::gDataOriginTRD, "TRDCFG", 0, framework::Lifetime::Condition}, *mTrapConfigEvent.get());
}
