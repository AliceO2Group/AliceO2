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

/// \file CalibratorConfigEvents.cxx
/// \brief Config Events calibrator
/// \author Sean Murray

#include "Framework/ProcessingContext.h"
#include "Framework/TimingInfo.h"
#include "Framework/InputRecord.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/MemFileHelper.h"

#include "DataFormatsTRD/Constants.h"
#include "TRDCalibration/CalibratorConfigEvents.h"
#include "DataFormatsTRD/TrapConfigEvent.h"
#include "TRDQC/StatusHelper.h"
#include "TRDBase/GeometryBase.h"

#include "TStopwatch.h"
#include <TFile.h>
#include <TTree.h>

#include <string>
#include <map>
#include <memory>
#include <ctime>

using namespace o2::trd::constants;

namespace o2::trd
{

void CalibratorConfigEvents::clearEventStructures()
{
  mCCDBObject.clear();
  for (auto& mcm : mTrapRegistersFrequencyMap) {
    for (auto& reg : mcm) {
      reg.clear();
    }
  }
  mTimesSeenMCM.fill(0);
  // mTimesSeenHCID.fill(0);
}

void CalibratorConfigEvents::init()
{
  std::unique_ptr<o2::ccdb::CCDBManagerInstance> mgr;
  mgr = std::make_unique<o2::ccdb::CCDBManagerInstance>("http://ccdb-test.cern.ch:8080");
  LOGP(info, "get valid half chambers from ccdb");
  auto now = std::chrono::high_resolution_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto timeStamp = now_ms.time_since_epoch();
  auto halfChamberStatus = mgr->getForTimeStamp<o2::trd::HalfChamberStatusQC>("TRD/Calib/HalfChamberStatusQC", timeStamp.count());
  if (halfChamberStatus == nullptr) {
    LOGP(info, "Could not find a halfchamberstatusqc for this time, searching for a known time");
    // TODO ...
  }
  //  setMaskedHalfChambers(halfChamberStatus->getBitSet());
}

void CalibratorConfigEvents::process(const gsl::span<MCMEvent>& MCMEvents)
{
  // if (mInitCompleted) {
  //   return;
  // }

  // set tree addresses
  if (mEnableOutput) {
    mOutTree->Branch("trapconfigevent", &mCCDBObject);
    //  what do we want to save?
  }

  // take incoming trapconfigevents and populate the array of array of maps, seen values for eventual
  // collapsing into a singular trapconfigevent at the end of the alloted accumulation time.
  // additionally fill in the mcmindex and the hcidpresent mcmpresent is built into the index.
  int count = 0;
  int mcmcount = 0;
  int mcmregmax = 23;
  int hcid = 0;
  uint32_t regvalue = 0;
  std::bitset<constants::MAXCHAMBER> hcidseen;
  // take the incoming events and pack them into the array of array of maps
  for (auto& mcmevent : MCMEvents) {
    for (int mcmreg = 0; mcmreg < TrapRegisters::kLastReg; ++mcmreg) {
      regvalue = mcmevent.getRegister(mcmreg, mCCDBObject.getRegisterInfo(mcmreg));
      auto mcmid = mcmevent.getMCMId();
      if (mcmid < constants::MAXMCMCOUNT) {
        mTimesSeenMCM[mcmid]++; // frequency map for values in the respective registers
        hcid = HelperMethods::getHCIDFromMCMId(mcmid);
        // LOGP(info, " we got hcid : {} from mcmid : {}", hcid, mcmid);
        if (!hcidseen.test(hcid)) {
          // first time for this hcid
          mTimesSeenHCID[hcid]++;
          hcidseen.set(hcid);
        }
        // LOGP(info, "XXX calibrator incoming value for mcmid #{} mcm register : [{},{}] value : {:08x} ", count++, mcmid, mCCDBObject.getRegisterName(mcmreg), mcmreg, regvalue);
        mCCDBObject.setRegisterValue(regvalue, mcmreg, mcmid);
        if (mTrapRegistersFrequencyMap[mcmid][mcmreg][regvalue] == 0) {
          //  this =1 is actually not required as ++ will increment the zero, its more here for clarity, as this is the case of regvalue not being in the map yet.
          mTrapRegistersFrequencyMap[mcmid][mcmreg][regvalue] = 1;
        } else {
          mTrapRegistersFrequencyMap[mcmid][mcmreg][regvalue]++;
          // LOGP(info, "XXXYYY  [{}]  for mcmid #{} mcm register : [{},{}] value : {:08x} count 1=={} ", count++, mcmid, mCCDBObject.getRegisterName(mcmreg), mcmreg, regvalue, mTrapRegistersFrequencyMap[mcmid][mcmreg].size());
        }
        if (mTrapRegistersFrequencyMap[mcmid][mcmreg].size() > 1) {
          int count = 0;
          for (auto& apair : mTrapRegistersFrequencyMap[mcmid][mcmreg]) {
            LOGP(debug, "XXX  [{}] GREATER than 1 value for mcmid #{} mcm register : [{},{}] value : {:08x} count 1=={} ", count++, mcmid, mCCDBObject.getRegisterName(mcmreg), mcmreg, apair.first, mTrapRegistersFrequencyMap[mcmid][mcmreg].size());
          }
        }
      } // if(mcmid<constants::MAXMCMCOUNT)
      else {
        LOGP(warn, "mcmid from mcmevent is too big : {} > {}", mcmid, constants::MAXMCMCOUNT);
      }
    }
  }
  mTimeReached++; // count the number of timeframes with configevents.
  LOGP(info, "increment time reached to {} ", mTimeReached);
}

bool CalibratorConfigEvents::hasEnoughData() const
{
  // determine if this slot has enough data ... normal calibration.
  // this method triggers a merge and a finaliseSlot.
  // if the slot does not have enough data this slot together with the next one are merged.

  // We therefore have 2 options :
  // a:keep all the difference and ergo, save to ccdb after each incoming config event.
  // b: the normal option, accumulate for 10,15 minutes and then compare and write to ccdb if new and flag qc.
  bool timereached = 0;
  if (mSaveAllChanges) {
    return true;
  } else {
    // case b:
    //  we just care about the time taken since the beginning.
    //
    if (mTimeReached % 10 && mTimeReached != 0) {
      LOGP(info, "Reached a count of 10 : {} ", mTimeReached);
      return true;
    } else {
      LOGP(info, "Not Reached a count of 10 : {} ", mTimeReached);
      return false;
    }
  }
}

void CalibratorConfigEvents::collapseRegisterValues()
{
  // map to store unique values for each register to infer mo
  std::array<std::map<uint32_t, uint32_t>, TrapRegisters::kLastReg> registersvaluemap;
  // Collapse the frequency maps of registers into singular values
  for (uint32_t mcmid = 0; mcmid < constants::MAXMCMCOUNT; ++mcmid) {

    if (mCCDBObject.isMCMPresent(mcmid)) {
      // LOGP(info, "Collapsing with mcmid {} is present", mcmid);
      auto mcmevent = mCCDBObject.getMCMEvent(mcmid);
      if (mTimesSeenMCM[mcmid] > 0) {
        // avoid those mcm that have no data.
        for (int mcmreg = 0; mcmreg < TrapRegisters::kLastReg; ++mcmreg) {
          // auto regvalue = mcmevent.getRegister(mcmreg,mCCDBObject.getTrapRegInfo(mcmreg));
          // do we have more than a single value?
          if (mTrapRegistersFrequencyMap[mcmid][mcmreg].size() > 1) {
            // find most frequent value
            auto maxelement = std::max_element(mTrapRegistersFrequencyMap[mcmid][mcmreg].begin(), mTrapRegistersFrequencyMap[mcmid][mcmreg].end(), [](const auto& x, const auto& y) {
              return x.second < y.second;
            });
            /*if(mcmreg== TrapRegisters::kADCMSK){
              LOGP(info, "mcm {} {} == {:08x} had {} values",mcmid,mCCDBObject.getRegisterName(mcmreg),maxelement->first,mTrapRegistersFrequencyMap[mcmid][mcmreg].size());
            }*/
            mCCDBObject.setRegisterValue(maxelement->first, mcmreg, mcmid);
            registersvaluemap[mcmreg][maxelement->first]++;
            int count = 0;
            for (auto& apair : mTrapRegistersFrequencyMap[mcmid][mcmreg]) {
              // debug output the multiple values and their frequency
            }
          } else {
            // we only have one value so use that one.
            //  if(mcmreg== TrapRegisters::kADCMSK){
            // LOGP(info, "mcm {} {} == {:08x} had {} values datacount {}",mcmid,mCCDBObject.getRegisterName(mcmreg), mTrapRegistersFrequencyMap[mcmid][mcmreg].begin()->first,mTrapRegistersFrequencyMap[mcmid][mcmreg].size(),mTrapRegistersFrequencyMap[mcmid][mcmreg].begin()->second);
            //  }
            auto data = mTrapRegistersFrequencyMap[mcmid][mcmreg].begin()->first;
            auto datacount = mTrapRegistersFrequencyMap[mcmid][mcmreg].begin()->second;
            mCCDBObject.setRegisterValue(data, mcmreg, mcmid);
            registersvaluemap[mcmreg][data]++;
          }
        }
      }
    } else {
      LOGP(info, "Collapsing with mcmid {} is not present", mcmid);
    }
  }

  // walk the register value map and figure out the most prevelant values for a particular register.
  for (int mcmreg = 0; mcmreg < TrapRegisters::kLastReg; ++mcmreg) {
    // auto& map = registervaluemap[mcmreg] ;
    auto mostcommonelement = std::max_element(registersvaluemap[mcmreg].begin(), registersvaluemap[mcmreg].end(), [](const auto& x, const auto& y) {
      return x.second < y.second;
    });
    // now to put the most common value in the default value for unenabled mcm for simulation
    mCCDBObject.setDefaultRegisterValue(mcmreg, mostcommonelement->first);
    LOGP(info, "register [{}:{}] has most common value of 0x{:08x} occured {} times, this reg had {} values in the map ", mcmreg, mCCDBObject.getRegisterName(mcmreg), mostcommonelement->first, mostcommonelement->second, registersvaluemap[mcmreg].size());
  }
}

void CalibratorConfigEvents::createFile()
{
  mEnableOutput = true;
  mOutFile = std::make_unique<TFile>("trd_configevents.root", "RECREATE");
  if (mOutFile->IsZombie()) {
    LOG(error) << "Failed to create output file for config events!";
    mEnableOutput = false;
    return;
  }
  mOutTree = std::make_unique<TTree>("calib", "Config Events");
  LOG(info) << "Created output file trd_configevents.root";
}

void CalibratorConfigEvents::closeFile()
{
  if (!mEnableOutput) {
    return;
  }

  try {
    mOutFile->cd();
    mOutTree->Write();
    mOutTree.reset();
    mOutFile->Close();
    mOutFile.reset();
  } catch (std::exception const& e) {
    LOG(error) << "Failed to write config event calibration data file, reason: " << e.what();
  }
  // after closing, we won't open a new file
  mEnableOutput = false;
}

bool CalibratorConfigEvents::isDifferent()
{
  // compare the accumulated config to that stored in the ccdb.
  return mCCDBObject.isConfigDifferent(mPreviousCCDBObject);
}

void CalibratorConfigEvents::stillMissingHCID(std::stringstream& missinghcid)
{
  // compare which hcid we see vs the ones that should be seen, send back the ones we should have seen but dont
  // std::array<uint32_t, constants::MAXHALFCHAMBER> mTimesSeenHCID; has the ones we have seen in config events
  missinghcid << "HCID seen in data but not in configs : ";
  for (uint32_t hcid = 0; hcid < constants::NCHAMBER * 2; ++hcid) {
    if (mHCIDSeenInData[hcid] > 0 && mTimesSeenHCID[hcid] == 0) {
      // HCID has data coming in but we did not get a config event on this hcid(link)
      missinghcid << fmt::format("[{} != {}], ", mHCIDSeenInData[hcid], mTimesSeenHCID[hcid]);
    }
  }
}

void CalibratorConfigEvents::stillMissingMCM(std::stringstream& missingmcm)
{
  // compare which mcm we see vs the ones that should be seen, send back the ones we should have seen but dont
  // std::array<uint32_t, constants::MAXMCMCOUNT> mTimesSeenMCM; has the ones we have seen in config events
  missingmcm << "MCM seen in data but not in configs : ";
  for (uint32_t mcmid = 0; mcmid < constants::NCHAMBER * 2; ++mcmid) {
    if (mMCMSeenInData[mcmid] > 0 && mTimesSeenMCM[mcmid] == 0) {
      // mcmid has data coming in but we did not get a config event on this mcmid
      missingmcm << fmt::format("[{} != {}], ", mMCMSeenInData[mcmid], mTimesSeenMCM[mcmid]);
    }
  }
}

} // namespace o2::trd
