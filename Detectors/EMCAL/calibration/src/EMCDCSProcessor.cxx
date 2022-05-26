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

/// \class EMCDCSProcessor
/// \brief DCS processor for EMCAL DPs
/// \ingroup EMCALcalib
/// \author Martin Poghosyan <martin.poghosyan@cern.ch>, ORNL
/// \since Sep 5th, 2021
///
/// processing of all DPs incuded (except of STU_TRU_ERROR counters)
/// improvemts expected:
/// 1. archive the configuration DPs only at DCS SOR
/// 2. include the processing of STU_TRU_ERROR counters
///
/// macro for reading the data from CCDB
/// O2/Detectors/EMCAL/calib/macros/readEMCALDCSentries.C
///
/// macro for creating the EMC aliases for DCS/Config:
/// O2/Detectors/EMCAL/calib/macros/makeEMCALCCDBEntryForDCS.C
/// They must match with those defined in PARA and in EMCALDCSDataProcessorSpec
///

#include <map>
#include <iterator>
#include "EMCALCalib/CalibDB.h"
#include "EMCALCalibration/EMCDCSProcessor.h"

using namespace o2::dcs;
using namespace o2::emcal;

using DeliveryType = o2::dcs::DeliveryType;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;

void EMCDCSProcessor::init(const std::vector<DPID>& pids)
{

  for (const auto& it : pids) {
    mPids[it] = false;
  }

  mFEECFG = std::make_unique<FeeDCS>();
  mELMB = std::make_unique<EMCELMB>();
  mELMBdata = std::make_unique<ElmbData>();

  mELMB->init();

  mTFprevELMB = 0;
}

int EMCDCSProcessor::process(const gsl::span<const DPCOM> dps)
{
  mUpdateFEEcfg = false;
  mUpdateELMB = false;

  if (mVerbose) {
    LOG(info) << "\n\n\nProcessing new TF\n-----------------";
  }

  for (auto& it : dps) {
    const auto& el = mPids.find(it.id);

    if (el == mPids.end()) {
      LOG(debug) << "DP " << it.id << " not found in the map of expected DPs- ignoring...";
      continue;
    } else {
      LOG(debug) << "DP " << it.id << " found in map";
    }

    processDP(it);
  }

  return 0;
}

void EMCDCSProcessor::processElmb()
{

  mUpdateELMB = true;
  mELMB->process();
  mELMBdata->setData(mELMB->getData());
  mELMB->reset();
}

int EMCDCSProcessor::processDP(const DPCOM& dp)
{

  auto& dpid = dp.id;
  const auto& type = dpid.get_type();
  auto& val = dp.data;

  if (mVerbose) {
    if (type == DPVAL_DOUBLE) {
      LOG(info) << "Processing DP = " << dp << ", with value = " << o2::dcs::getValue<double>(dp);
    } else if (type == DPVAL_INT) {
      LOG(info) << "Processing DP = " << dp << ", with value = " << o2::dcs::getValue<int32_t>(dp);
    } else if (type == DPVAL_UINT) {
      LOG(info) << "Processing DP = " << dp << ", with value = " << o2::dcs::getValue<uint32_t>(dp);
    }
  }

  if ((type == DPVAL_INT) || (type == DPVAL_UINT)) // FEE config params and STU_TRU error counters
  {
    auto& dpval_prev = mapFEEcfg[dpid];
    if (dpval_prev.size() == 0 || val.get_epoch_time() != dpval_prev.back().get_epoch_time()) // compare the time stamps
    {
      dpval_prev.push_back(val); // do we need to archive them all???
                                 //     mUpdateFEEcfg = true; FEE data will be updated based on SOR/EOR

      FillFeeDP(dp);
      setTF(val.get_epoch_time()); // fix: this must not be here!
    }
  } else if (type == DPVAL_DOUBLE) { // ELMB data
    FillElmbDP(dp);
  }
  // printPDCOM(dp);

  return 0;
}

void EMCDCSProcessor::FillElmbDP(const DPCOM& dpcom)
{
  auto& dpid = dpcom.id;
  const auto type = dpid.get_type();
  std::string type_str = show(type);

  std::string alias(dpid.get_alias());

  auto& dpval = dpcom.data;
  auto val = o2::dcs::getValue<double>(dpcom); // dpval.payload_pt1;

  std::size_t index;
  int iPT = -1;

  if ((index = alias.find("EMC_PT")) != std::string::npos) {
    std::sscanf(alias.data(), "EMC_PT_%d.Temperature", &iPT);
    LOG(debug) << "alias=" << alias.data() << ":  iPT=" << iPT << ", val=" << val;

    if (iPT < 0 || iPT > 159) {
      LOG(error) << "Wrong Sensor Index iPT=" << iPT << " for DP " << alias.data();
      return;
    }
    mELMB->addMeasurement(iPT, val);
  } else {
    LOG(info) << "EMC_PT pattern not found for DPype = DPVAL_DOUBLE: alias = " << alias.data();
  }
  return;
}

void EMCDCSProcessor::FillFeeDP(const DPCOM& dpcom)
{
  auto& dpid = dpcom.id;
  const auto type = dpid.get_type();
  std::string type_str = show(type);

  std::string alias(dpid.get_alias());

  auto& dpval = dpcom.data;
  auto ts = dpval.get_epoch_time();
  auto val = dpval.payload_pt1;

  std::size_t index;
  int iTRU = -1;
  int iMask = -1;
  int iSM = -1;

  //  if (mVerbose) {
  //    LOG(info) << "EMCDCSProcessor::FillFeeDP called";
  //  }
  //

  if ((index = alias.find("STU_ERROR_COUNT_TRU")) != std::string::npos) {
    // processing of STU_TRU error counters not included yet
    return;
  }

  else if (alias.find("EMC_RUNNUMBER") != std::string::npos) {
    if (mFEECFG->getRunNumber() != val) {
      mUpdateFEEcfg = true;
    }
    if (mRunNumberFromGRP == -2) { // no run number from GRP
      mFEECFG->setRunNumber(val);
    } else {
      mFEECFG->setRunNumber(mRunNumberFromGRP);
      if (mRunNumberFromGRP != val) {
        LOG(error) << "RunNumber from GRP (=" << mRunNumberFromGRP << ") and from EMC DCS (=" << val << ") are not consistant";
      }
    }
  } else if (alias.find("EMC_DDL_LIST0") != std::string::npos) {
    mFEECFG->setDDLlist0(val);
  } else if (alias.find("EMC_DDL_LIST1") != std::string::npos) {
    mFEECFG->setDDLlist1(val);
  } else if ((index = alias.find("SRU")) != std::string::npos) {
    if ((index = alias.find("FMVER")) != std::string::npos) {
      std::sscanf((std::string(alias.substr(index - 3, 2))).data(), "%02d", &iSM);
      if (iSM < 0 || iSM >= 20) {
        LOG(error) << "ERROR : iSM = " << iSM << " for" << alias.data();
        return;
      }
      mFEECFG->setSRUFWversion(iSM, val);
    } else if ((index = alias.find("CFG")) != std::string::npos) {
      std::sscanf((std::string(alias.substr(index - 3, 2))).data(), "%02d", &iSM);
      if (iSM < 0 || iSM >= 20) {
        LOG(error) << "ERROR : iSM = " << iSM << " for" << alias.data();
        return;
      }
      mFEECFG->setSRUconfig(iSM, val);
    }
  } else if ((index = alias.find("TRU")) != std::string::npos) {
    std::sscanf((std::string(alias.substr(index + 3, 2))).data(), "%02d", &iTRU);

    if (iTRU < 0 || iTRU >= kNTRU) {
      LOG(error) << "ERROR : iTRU = " << iTRU << " for" << alias.data();
      return;
    }

    mTRU = mFEECFG->getTRUDCS(iTRU);
    if (alias.find("L0ALGSEL") != std::string::npos) {
      mTRU.setL0SEL(val);
    } else if (alias.find("PEAKFINDER") != std::string::npos) {
      mTRU.setSELPF(val);
    } else if (alias.find("GLOBALTHRESH") != std::string::npos) {
      mTRU.setGTHRL0(val);
    } else if (alias.find("COSMTHRESH") != std::string::npos) {
      mTRU.setL0COSM(val);
    } else if ((index = alias.find("MASK")) != std::string::npos) {
      std::sscanf((std::string(alias.substr(index + 4, 1))).data(), "%02d", &iMask);
      mTRU.setMaskReg(val, iMask);
      if (iMask < 0 || iMask > 5) {
        LOG(error) << "ERROR : iMask = " << iMask << " for" << alias.data();
        return;
      }
    }

    mFEECFG->setTRUDCS(iTRU, mTRU);
  } else if ((index = alias.find("_STU_")) != std::string::npos) {
    bool kEMC = (std::string(alias.substr(index - 3, 3))).compare(0, 3, "DMC", 0, 3);
    mSTU = kEMC ? mFEECFG->getSTUDCSEMCal() : mFEECFG->getSTUDCSDCal();

    if (alias.find("MEDIAN") != std::string::npos) {
      mSTU.setMedianMode(val);
    } else if (alias.find("GETRAW") != std::string::npos) {
      mSTU.setRawData(val);
    } else if (alias.find("REGION") != std::string::npos) {
      mSTU.setRegion(val);
    } else if (alias.find("FWVERS") != std::string::npos) {
      mSTU.setFw(val);
    } else if (alias.find("PATCHSIZE") != std::string::npos) {
      mSTU.setPatchSize(val);
    } else if ((index = alias.find("STU_G")) != std::string::npos) {
      char par1;
      int par2;
      std::sscanf((std::string(alias.substr(index + 5, 2))).data(), "%c%d", &par1, &par2);
      if (par2 == 0) {
        mSTU.setGammaHigh((int)par1 - 65, val);
      } else {
        mSTU.setGammaLow((int)par1 - 65, val);
      }
    } else if ((index = alias.find("STU_J")) != std::string::npos) {
      char par1;
      int par2;
      std::sscanf((std::string(alias.substr(index + 5, 2))).data(), "%c%d", &par1, &par2);
      if (par2 == 0) {
        mSTU.setJetHigh((int)par1 - 65, val);
      } else {
        mSTU.setJetLow((int)par1 - 65, val);
      }
    }

    if (kEMC) {
      mFEECFG->setSTUEMCal(mSTU);
    } else {
      mFEECFG->setSTUDCal(mSTU);
    }
  }
}

void EMCDCSProcessor::updateElmbCCDBinfo()
{
  if (mVerbose) {
    LOG(info) << "updating Temperture objects in CCDB";
    //    LOG(info) << "Temperture objects to be written in CCDB\n";
    //              << *mFEECFG.get();
  }

  std::map<std::string, std::string> metadata;
  metadata["responsible"] = "Martin Poghosyan";
  prepareCCDBobjectInfo(*mELMBdata.get(), mccdbELMBinfo, o2::emcal::CalibDB::getCDBPathTemperatureSensor(), mTF, metadata);
}

void EMCDCSProcessor::updateFeeCCDBinfo()
{
  if (mVerbose) {
    LOG(info) << "updating FEE DCS objects  in CCDB";
    //    LOG(info) << "FEE DCS object to be written in CCDB\n"
    //              << *mFEECFG.get();
  }
  std::map<std::string, std::string> metadata;
  metadata["responsible"] = "Martin Poghosyan";
  prepareCCDBobjectInfo(*mFEECFG.get(), mccdbFEEcfginfo, o2::emcal::CalibDB::getCDBPathFeeDCS(), mTF, metadata);
}

void EMCDCSProcessor::printPDCOM(const DPCOM& dpcom)
{
  auto& dpid = dpcom.id;
  const auto type = dpid.get_type();
  std::string type_str = show(type);

  auto alias = dpid.get_alias();

  auto& dpval = dpcom.data;
  auto ts = dpval.get_epoch_time();
  auto val = dpval.payload_pt1;

  std::cout << "DPCOM Info:";

  std::cout << " alias: " << alias;
  std::cout << " | type : " << type_str;
  std::cout << " | ts   : " << ts;
  std::cout << " | value: " << val << std::endl;
}
