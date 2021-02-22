// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDWorkflow/TRDTrapSimulatorSpec.h"

#include <cstdlib>
#include <thread> // to detect number of hardware threads
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>
#include <unistd.h> // for getppid
#include <chrono>
#include <gsl/span>
#include <iostream>
#include <fstream>

#include "TChain.h"
#include "TFile.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include "fairlogger/Logger.h"
#include "CCDB/BasicCCDBManager.h"

#include "DataFormatsParameters/GRPObject.h"
#include "TRDBase/Digit.h"
#include "TRDBase/Calibrations.h"
#include "TRDSimulation/TrapSimulator.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/Constants.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

using namespace constants;

TrapConfig* TRDDPLTrapSimulatorTask::getTrapConfig()
{
  // return an existing TRAPconfig or load it from the CCDB
  // in case of failure, a default TRAPconfig is created
  LOG(debug) << "start of gettrapconfig";
  if (mTrapConfig) {
    LOG(debug) << "mTrapConfig is valid : 0x" << std::hex << mTrapConfig << std::dec;
    return mTrapConfig;
  } else {
    LOG(debug) << "mTrapConfig is invalid : 0x" << std::hex << mTrapConfig << std::dec;

    //// bypass pulling in the traditional default trapconfigs from ccdb, will sort out latert
    // try to load the requested configuration
    loadTrapConfig();
    //calib.
    if (mTrapConfig->getConfigName() == "" && mTrapConfig->getConfigVersion() == "") {
      //some trap configs dont have config name and version set, in those cases, just show the file name used.
      LOG(info) << "using TRAPconfig :\"" << mTrapConfigName;
    } else {
      LOG(info) << "using TRAPconfig :\"" << mTrapConfig->getConfigName().c_str() << "\".\"" << mTrapConfig->getConfigVersion().c_str() << "\"";
    }
    // we still have to load the gain tables
    // if the gain filter is active
    return mTrapConfig;
  } // end of else from if mTrapConfig
}

void TRDDPLTrapSimulatorTask::loadDefaultTrapConfig()
{
  //this loads a trap config from a root file for those times when the ccdb is not around and you want to keep working.
  TFile* f;
  f = new TFile("DefaultTrapConfig.root");
  mTrapConfig = (o2::trd::TrapConfig*)f->Get("ccdb_object");
  if (mTrapConfig == nullptr) {
    LOG(fatal) << "failed to load from ccdb, and attempted to load from disk, you seem to be really out of luck.";
  }
  // else we have loaded the trap config successfully.
}
void TRDDPLTrapSimulatorTask::loadTrapConfig()
{
  // try to load the specified configuration from the CCDB

  LOG(info) << "looking for TRAPconfig " << mTrapConfigName;

  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbmgr.setTimestamp(mRunNumber);
  //default is : mTrapConfigName="cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549";
  mTrapConfigName = "c"; //cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549";
  mTrapConfig = ccdbmgr.get<o2::trd::TrapConfig>("TRD_test/TrapConfig2020/" + mTrapConfigName);
  if (mTrapConfig == nullptr) {
    //failed to find or open or connect or something to get the trapconfig from the ccdb.
    //first check the directory listing.
    LOG(warn) << " failed to get trapconfig from ccdb with name :  " << mTrapConfigName;
    loadDefaultTrapConfig();
  } else {
    //TODO figure out how to get the debug level from logger and only do this for debug option to --severity debug (or what ever the command actualy is)
    if (mEnableTrapConfigDump) {
      mTrapConfig->DumpTrapConfig2File("run3trapconfig_dump");
    }
  }
}

void TRDDPLTrapSimulatorTask::setOnlineGainTables()
{
  //check FGBY from trapconfig.
  //check the input parameter of trd-onlinegaincorrection.
  //warn if you have chosen a trapconfig with gaincorrections but chosen not to use them.
  if (mEnableOnlineGainCorrection) {
    if (mTrapConfig->getTrapReg(TrapConfig::kFGBY) == 0) {
      LOG(warn) << "you have asked to do online gain calibrations but the selected trap config does not have FGBY enabled, so modifying trapconfig to conform to your command line request. OnlineGains will be 1, i.e. no effect.";
      for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
        mTrapConfig->setTrapReg(TrapConfig::kFGBY, 1, iDet);
      }
    }
    mCalib->setOnlineGainTables(mOnlineGainTableName);
    //TODO add some error checking inhere.
    // gain factors are per MCM
    // allocate the registers accordingly
    for (int ch = 0; ch < NADCMCM; ++ch) {
      TrapConfig::TrapReg_t regFGAN = (TrapConfig::TrapReg_t)(TrapConfig::kFGA0 + ch);
      TrapConfig::TrapReg_t regFGFN = (TrapConfig::TrapReg_t)(TrapConfig::kFGF0 + ch);
      mTrapConfig->setTrapRegAlloc(regFGAN, TrapConfig::kAllocByMCM);
      mTrapConfig->setTrapRegAlloc(regFGFN, TrapConfig::kAllocByMCM);
    }

    for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
      const int nRobs = Geometry::getStack(iDet) == 2 ? Geometry::ROBmaxC0() : Geometry::ROBmaxC1();
      for (int rob = 0; rob < nRobs; ++rob) {
        for (int mcm = 0; mcm < NMCMROB; ++mcm) {
          // set ADC reference voltage
          mTrapConfig->setTrapReg(TrapConfig::kADCDAC, mCalib->getOnlineGainAdcdac(iDet, rob, mcm), iDet, rob, mcm);
          // set constants channel-wise
          for (int ch = 0; ch < NADCMCM; ++ch) {
            TrapConfig::TrapReg_t regFGAN = (TrapConfig::TrapReg_t)(TrapConfig::kFGA0 + ch);
            TrapConfig::TrapReg_t regFGFN = (TrapConfig::TrapReg_t)(TrapConfig::kFGF0 + ch);
            mTrapConfig->setTrapReg(regFGAN, mCalib->getOnlineGainFGAN(iDet, rob, mcm, ch), iDet, rob, mcm);
            mTrapConfig->setTrapReg(regFGFN, mCalib->getOnlineGainFGFN(iDet, rob, mcm, ch), iDet, rob, mcm);
          }
        }
      }
    }
  } else if (mTrapConfig->getTrapReg(TrapConfig::kFGBY) == 1) {
    LOG(warn) << "you have asked to not use online gain calibrations but the selected trap config does have FGBY enabled. Gain calibrations will be 1, i.e. no effect";
  }
}

void TRDDPLTrapSimulatorTask::init(o2::framework::InitContext& ic)
{
  LOG(debug) << "entering init";
  mPrintTrackletOptions = ic.options().get<int>("trd-printtracklets");
  mDrawTrackletOptions = ic.options().get<int>("trd-drawtracklets");
  mShowTrackletStats = ic.options().get<int>("show-trd-trackletstats");
  mTrapConfigName = ic.options().get<std::string>("trd-trapconfig");
  mDebugRejectedTracklets = ic.options().get<bool>("trd-debugrejectedtracklets");
  mEnableOnlineGainCorrection = ic.options().get<bool>("trd-onlinegaincorrection");
  mOnlineGainTableName = ic.options().get<std::string>("trd-onlinegaintable");
  mRunNumber = ic.options().get<int>("trd-runnum");
  mEnableTrapConfigDump = ic.options().get<bool>("trd-dumptrapconfig");
  //Connect to CCDB for all things needing access to ccdb, trapconfig and online gains
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  mCalib = std::make_unique<Calibrations>();
  mCalib->setCCDBForSimulation(mRunNumber);
  getTrapConfig();
  setOnlineGainTables();
  LOG(info) << "Trap Simulator Device initialised for config : " << mTrapConfigName;
}

void TRDDPLTrapSimulatorTask::run(o2::framework::ProcessingContext& pc)
{
  LOG(info) << "TRD Trap Simulator Device running over incoming message";

  // get inputs for the TrapSimulator
  // the digits are going to be sorted, we therefore need a copy of the vector rather than an object created
  // directly on the input data, the output vector however is created directly inside the message
  // memory thus avoiding copy by snapshot

  /*********
   * iNPUTS
   ********/

  auto inputDigits = pc.inputs().get<gsl::span<o2::trd::Digit>>("digitinput");
  auto digitMCLabels = pc.inputs().get<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>("labelinput");
  auto inputTriggerRecords = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("triggerrecords");

  if (inputDigits.size() == 0 || inputTriggerRecords.size() == 0) {
    LOG(WARNING) << "Did not receive any digits, trigger records, or neither one nor the other. Aborting.";
    return;
  }

  /* *****
   * setup data objects
   * *****/

  // trigger records to index the 64bit tracklets.yy
  std::vector<o2::trd::TriggerRecord> trackletTriggerRecords(inputTriggerRecords.begin(), inputTriggerRecords.end()); // copy over the whole thing but we only really want the bunch crossing info.
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> trackletMCLabels;
  //index of digits, TODO refactor to a digitindex class.
  std::vector<unsigned int> digitIndices(inputDigits.size());
  //set up structures to hold the returning tracklets.
  std::vector<Tracklet64> trapTrackletsAccum;

  /* *******
   * reserve sizes
   * *******/

  LOG(debug) << "Read in digits with size of : " << inputDigits.size() << ". Labels contain : " << digitMCLabels.getNElements() << " elements with and index size of  : " << digitMCLabels.getIndexedSize() << " and triggerrecord count of :" << inputTriggerRecords.size();
  if (digitMCLabels.getIndexedSize() != inputDigits.size()) {
    LOG(warn) << "Digits and Labels coming into TrapSimulator are of differing sizes, labels will be jibberish. " << digitMCLabels.getIndexedSize() << "!=" << inputDigits.size();
  }
  trapTrackletsAccum.reserve(inputDigits.size() / 500);

  //Build the digits index.
  std::iota(digitIndices.begin(), digitIndices.end(), 0);
  //sort the digits array TODO refactor this intoa vector index sort and possibly generalise past merely digits.
  auto sortstart = std::chrono::high_resolution_clock::now();
  // TODO check if sorting is still needed
  for (auto& trig : inputTriggerRecords) {
    std::stable_sort(std::begin(digitIndices) + trig.getFirstEntry(), std::begin(digitIndices) + trig.getNumberOfObjects() + trig.getFirstEntry(),
                     [&inputDigits](unsigned int i, unsigned int j) { return inputDigits[i].getDetector() < inputDigits[j].getDetector(); });
  }

  mSortingTime = std::chrono::high_resolution_clock::now() - sortstart;
  LOG(warn) << "TRD Digit Sorting took " << mSortingTime.count();

  auto timeDigitLoopStart = std::chrono::high_resolution_clock::now();

  int currDetector = -1;

  for (int iTrig = 0; iTrig < inputTriggerRecords.size(); ++iTrig) {
    int nTrackletsInTrigRec = 0;
    for (int iDigit = inputTriggerRecords[iTrig].getFirstEntry(); iDigit < (inputTriggerRecords[iTrig].getFirstEntry() + inputTriggerRecords[iTrig].getNumberOfObjects()); ++iDigit) {
      const auto& digit = &inputDigits[digitIndices[iDigit]];
      if (currDetector < 0) {
        currDetector = digit->getDetector();
      }
      if (currDetector != digit->getDetector()) {
        // we switch to a new chamber, process all TRAPs of the last chamber which contain data
        int currStack = (currDetector % NCHAMBERPERSEC) / NLAYER;
        int nTrapsMax = (currStack == 2) ? NROBC0 * NMCMROB : NROBC1 * NMCMROB;
        for (int iTrap = 0; iTrap < nTrapsMax; ++iTrap) {
          if (!mTrapSimulator[iTrap].isDataSet()) {
            continue;
          }
          auto timeTrapProcessingStart = std::chrono::high_resolution_clock::now();
          mTrapSimulator[iTrap].filter();
          mTrapSimulator[iTrap].tracklet();
          mTrapSimAccumulatedTime += std::chrono::high_resolution_clock::now() - timeTrapProcessingStart;
          nTrackletsInTrigRec += mTrapSimulator[iTrap].getTrackletArray64().size();
          trapTrackletsAccum.insert(trapTrackletsAccum.end(), mTrapSimulator[iTrap].getTrackletArray64().begin(), mTrapSimulator[iTrap].getTrackletArray64().end());
          // TODO get and output MC labels
          mTrapSimulator[iTrap].reset();
        }
        currDetector = digit->getDetector();
      }
      int trapIdx = digit->getROB() * NMCMROB + digit->getMCM();
      if (!mTrapSimulator[trapIdx].isDataSet()) {
        mTrapSimulator[trapIdx].init(mTrapConfig, digit->getDetector(), digit->getROB(), digit->getMCM());
      }
      // TODO check MC label part
      std::vector<o2::MCCompLabel> dummyLabels;
      /*
      auto digitslabels = digitMCLabels.getLabels(iDigit);
      for (auto& tmplabel : digitslabels) {
        tmplabels.push_back(tmplabel);
      }
      */
      // end MC label part
      mTrapSimulator[trapIdx].setData(digit->getChannel(), digit->getADC(), dummyLabels);
    }
    // take care of the TRAPs for the last chamber
    int currStack = (currDetector % NCHAMBERPERSEC) / NLAYER;
    int nTrapsMax = (currStack == 2) ? NROBC0 * NMCMROB : NROBC1 * NMCMROB;
    for (int iTrap = 0; iTrap < nTrapsMax; ++iTrap) {
      if (!mTrapSimulator[iTrap].isDataSet()) {
        continue;
      }
      auto timeTrapProcessingStart = std::chrono::high_resolution_clock::now();
      mTrapSimulator[iTrap].filter();
      mTrapSimulator[iTrap].tracklet();
      mTrapSimAccumulatedTime += std::chrono::high_resolution_clock::now() - timeTrapProcessingStart;
      nTrackletsInTrigRec += mTrapSimulator[iTrap].getTrackletArray64().size();
      trapTrackletsAccum.insert(trapTrackletsAccum.end(), mTrapSimulator[iTrap].getTrackletArray64().begin(), mTrapSimulator[iTrap].getTrackletArray64().end());
      // TODO get and output MC labels
      mTrapSimulator[iTrap].reset();
    }
    trackletTriggerRecords[iTrig].setDataRange(trapTrackletsAccum.size() - nTrackletsInTrigRec, nTrackletsInTrigRec);
    currDetector = -1;
  }

  LOG(info) << "Trap simulator found " << trapTrackletsAccum.size() << " tracklets from " << inputDigits.size() << " Digits and " << trackletMCLabels.getIndexedSize() << " associated MC Label indexes and " << trackletMCLabels.getNElements() << " associated MC Labels";
  if (mShowTrackletStats > 0) {
    mDigitLoopTime = std::chrono::high_resolution_clock::now() - timeDigitLoopStart;
    LOG(info) << "Trap Simulator done ";
    LOG(info) << "Digit loop took : " << mDigitLoopTime.count() << "s";
    LOG(info) << "TRAP processing (filter+tracklet) took : " << mTrapSimAccumulatedTime.count() << "s";
  }
  LOG(debug) << "END OF RUN .............";
  pc.outputs().snapshot(Output{"TRD", "TRACKLETS", 0, Lifetime::Timeframe}, trapTrackletsAccum);
  pc.outputs().snapshot(Output{"TRD", "TRKTRGRD", 0, Lifetime::Timeframe}, trackletTriggerRecords);
  //pc.outputs().snapshot(Output{"TRD", "TRKLABELS", 0, Lifetime::Timeframe}, trackletMCLabels);

  LOG(debug) << "exiting the trap sim run method ";
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec()
{
  return DataProcessorSpec{"TRAP", Inputs{InputSpec{"digitinput", "TRD", "DIGITS", 0}, InputSpec{"triggerrecords", "TRD", "TRGRDIG", 0}, InputSpec{"labelinput", "TRD", "LABELS", 0}},

                           Outputs{OutputSpec{"TRD", "TRACKLETS", 0, Lifetime::Timeframe}, // this is the 64 tracklet words
                                   OutputSpec{"TRD", "TRKTRGRD", 0, Lifetime::Timeframe},
                                   /*OutputSpec{"TRD", "TRKDIGITS", 0, Lifetime::Timeframe},*/
                                   OutputSpec{"TRD", "TRKLABELS", 0, Lifetime::Timeframe}},
                           AlgorithmSpec{adaptFromTask<TRDDPLTrapSimulatorTask>()},
                           Options{
                             {"show-trd-trackletstats", VariantType::Int, 25000, {"Display the accumulated size and capacity at number of track intervals"}},
                             {"trd-trapconfig", VariantType::String, "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549", {"Name of the trap config from the CCDB default:cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549"}},
                             {"trd-drawtracklets", VariantType::Int, 0, {"Bitpattern of input to TrapSimulator Draw method one histogram per chip not per tracklet, 1=raw,2=hits,4=tracklets, 7 for all"}},
                             {"trd-printtracklets", VariantType::Int, 0, {"Bitpattern of input to TrapSimulator print method, "}},
                             {"trd-onlinegaincorrection", VariantType::Bool, false, {"Apply online gain calibrations, mostly for back checking to run2 by setting FGBY to 0"}},
                             {"trd-onlinegaintable", VariantType::String, "Krypton_2015-02", {"Online gain table to be use, names found in CCDB, obviously trd-onlinegaincorrection must be set as well."}},
                             {"trd-debugrejectedtracklets", VariantType::Bool, false, {"Output all MCM where tracklets were not identified"}},
                             {"trd-dumptrapconfig", VariantType::Bool, false, {"Dump the selected trap configuration at loading time, to text file"}},
                             {"trd-runnum", VariantType::Int, 297595, {"Run number to use to anchor simulation to, defaults to 297595"}}}};
};

} //end namespace trd
} //end namespace o2
