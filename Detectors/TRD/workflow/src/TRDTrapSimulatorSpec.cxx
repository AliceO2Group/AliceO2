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

#include "TRDWorkflow/TRDTrapSimulatorSpec.h"

#include <chrono>
#include <optional>
#include <gsl/span>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

#include "TFile.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "fairlogger/Logger.h"
#include "CCDB/BasicCCDBManager.h"

#include "TRDBase/Calibrations.h"
#include "TRDSimulation/TRDSimParams.h"
#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/TriggerRecord.h"

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

void TRDDPLTrapSimulatorTask::processTRAPchips(int& nTracklets, std::vector<Tracklet64>& trackletsAccum, std::array<TrapSimulator, NMCMHCMAX>& trapSimulators, std::vector<short>& digitCounts, std::vector<int>& digitIndices)
{
  // TRAP processing for current half chamber
  for (int iTrap = 0; iTrap < NMCMHCMAX; ++iTrap) {
    if (!trapSimulators[iTrap].isDataSet()) {
      continue;
    }
    trapSimulators[iTrap].filter();
    trapSimulators[iTrap].tracklet();
    auto trackletsOut = trapSimulators[iTrap].getTrackletArray64();
    nTracklets += trackletsOut.size();
    trackletsAccum.insert(trackletsAccum.end(), trackletsOut.begin(), trackletsOut.end());
    if (mUseMC) {
      auto digitCountOut = trapSimulators[iTrap].getTrackletDigitCount();
      digitCounts.insert(digitCounts.end(), digitCountOut.begin(), digitCountOut.end());
      auto digitIndicesOut = trapSimulators[iTrap].getTrackletDigitIndices();
      digitIndices.insert(digitIndices.end(), digitIndicesOut.begin(), digitIndicesOut.end());
    }
    trapSimulators[iTrap].reset();
  }
}

void TRDDPLTrapSimulatorTask::init(o2::framework::InitContext& ic)
{
  mTrapConfigName = ic.options().get<std::string>("trd-trapconfig");
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
#ifdef WITH_OPENMP
  int askedThreads = TRDSimParams::Instance().digithreads;
  int maxThreads = omp_get_max_threads();
  if (askedThreads < 0) {
    mNumThreads = maxThreads;
  } else {
    mNumThreads = std::min(maxThreads, askedThreads);
  }
  LOG(info) << "Trap simulation running with " << mNumThreads << " threads ";
#endif
  LOG(info) << "Trap Simulator Device initialised for config : " << mTrapConfigName;
}

void TRDDPLTrapSimulatorTask::run(o2::framework::ProcessingContext& pc)
{
  // this method steeres the processing of the TRAP simulation
  LOG(info) << "TRD Trap Simulator Device running over incoming message";

  // input
  auto digits = pc.inputs().get<gsl::span<o2::trd::Digit>>("digitinput");                       // block of TRD digits
  auto triggerRecords = pc.inputs().get<std::vector<o2::trd::TriggerRecord>>("triggerrecords"); // time and number of digits for each collision
  LOG(debug) << "Received " << digits.size() << " digits from " << triggerRecords.size() << " collisions";

  // load MC information if label processing is enabeld
  const o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>* lblDigitsPtr = nullptr;
  using lblType = std::decay_t<decltype(pc.inputs().get<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(""))>;
  std::optional<lblType> lblDigits;
  if (mUseMC) {
    lblDigits.emplace(pc.inputs().get<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>("labelinput")); // MC labels associated to the input digits
    lblDigitsPtr = &lblDigits.value();
    LOG(debug) << "Labels contain " << lblDigitsPtr->getNElements() << " elements with and indexed size of " << lblDigitsPtr->getIndexedSize();
    if (lblDigitsPtr->getIndexedSize() != digits.size()) {
      LOG(warn) << "Digits and Labels coming into TrapSimulator are of differing sizes, labels will be jibberish. " << lblDigitsPtr->getIndexedSize() << "!=" << digits.size();
    }
  }

  // output
  std::vector<Tracklet64> tracklets;                               // calculated tracklets
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> lblTracklets; // MC labels for the tracklets, taken from the digits which make up the tracklet (duplicates are removed)

  auto timeProcessingStart = std::chrono::high_resolution_clock::now(); // measure total processing time

  // sort digits by half chamber ID for each collision and keep track in index vector
  auto sortStart = std::chrono::high_resolution_clock::now();
  std::vector<unsigned int> digitIdxArray(digits.size()); // digit indices sorted by half chamber ID for each time frame
  std::iota(digitIdxArray.begin(), digitIdxArray.end(), 0);
  for (auto& trig : triggerRecords) {
    std::stable_sort(std::begin(digitIdxArray) + trig.getFirstDigit(), std::begin(digitIdxArray) + trig.getNumberOfDigits() + trig.getFirstDigit(),
                     [&digits](unsigned int i, unsigned int j) { return digits[i].getHCId() < digits[j].getHCId(); });
  }
  auto sortTime = std::chrono::high_resolution_clock::now() - sortStart;

  // prepare data structures for accumulating results per collision
  std::vector<int> nTracklets(triggerRecords.size());
  std::vector<std::vector<Tracklet64>> trackletsAccum;
  trackletsAccum.resize(triggerRecords.size());
  std::vector<std::vector<short>> digitCountsAccum; // holds the number of digits included in each tracklet (therefore has the same number of elements as trackletsAccum)
  // digitIndicesAccum holds the global indices of the digits which comprise the tracklets
  // with the help of digitCountsAccum one can loop through this vector and find the corresponding digit indices for each tracklet
  std::vector<std::vector<int>> digitIndicesAccum;
  digitCountsAccum.resize(triggerRecords.size());
  digitIndicesAccum.resize(triggerRecords.size());

  auto timeParallelStart = std::chrono::high_resolution_clock::now();

#ifdef WITH_OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(mNumThreads)
#endif
  for (int iTrig = 0; iTrig < triggerRecords.size(); ++iTrig) {
    int currHCId = -1;
    std::array<TrapSimulator, NMCMHCMAX> trapSimulators{}; //the up to 64 trap simulators for a single half chamber
    for (int iDigit = triggerRecords[iTrig].getFirstDigit(); iDigit < (triggerRecords[iTrig].getFirstDigit() + triggerRecords[iTrig].getNumberOfDigits()); ++iDigit) {
      const auto& digit = &digits[digitIdxArray[iDigit]];
      if (currHCId < 0) {
        currHCId = digit->getHCId();
      }
      if (currHCId != digit->getHCId()) {
        // we switch to a new half chamber, process all TRAPs of the previous half chamber which contain data
        processTRAPchips(nTracklets[iTrig], trackletsAccum[iTrig], trapSimulators, digitCountsAccum[iTrig], digitIndicesAccum[iTrig]);
        currHCId = digit->getHCId();
      }
      // fill the digit data into the corresponding TRAP chip
      int trapIdx = (digit->getROB() / 2) * NMCMROB + digit->getMCM();
      if (!trapSimulators[trapIdx].isDataSet()) {
        trapSimulators[trapIdx].init(mTrapConfig, digit->getDetector(), digit->getROB(), digit->getMCM());
      }
      trapSimulators[trapIdx].setData(digit->getChannel(), digit->getADC(), digitIdxArray[iDigit]);
    }
    // take care of the TRAPs for the last half chamber
    processTRAPchips(nTracklets[iTrig], trackletsAccum[iTrig], trapSimulators, digitCountsAccum[iTrig], digitIndicesAccum[iTrig]);
  } // done with parallel processing
  auto parallelTime = std::chrono::high_resolution_clock::now() - timeParallelStart;

  // accumulate results and add MC labels
  for (int iTrig = 0; iTrig < triggerRecords.size(); ++iTrig) {
    if (mUseMC) {
      int currDigitIndex = 0; // counter for all digits which are associated to tracklets
      int trkltIdxStart = tracklets.size();
      for (int iTrklt = 0; iTrklt < nTracklets[iTrig]; ++iTrklt) {
        int tmp = currDigitIndex;
        for (int iDigitIndex = tmp; iDigitIndex < tmp + digitCountsAccum[iTrig][iTrklt]; ++iDigitIndex) {
          if (iDigitIndex == tmp) {
            // for the first digit composing the tracklet we don't need to check for duplicate labels
            lblTracklets.addElements(trkltIdxStart + iTrklt, lblDigitsPtr->getLabels(digitIndicesAccum[iTrig][iDigitIndex]));
          } else {
            // in case more than one digit composes the tracklet we add only the labels
            // from the additional digit(s) which are not already contained in the previous
            // digit(s)
            auto currentLabels = lblTracklets.getLabels(trkltIdxStart + iTrklt);
            auto newLabels = lblDigitsPtr->getLabels(digitIndicesAccum[iTrig][iDigitIndex]);
            for (const auto& newLabel : newLabels) {
              bool alreadyIn = false;
              for (const auto& currLabel : currentLabels) {
                if (currLabel.compare(newLabel)) {
                  alreadyIn = true;
                  break;
                }
              }
              if (!alreadyIn) {
                lblTracklets.addElement(trkltIdxStart + iTrklt, newLabel);
              }
            }
          }
          ++currDigitIndex;
        }
      }
    }
    tracklets.insert(tracklets.end(), trackletsAccum[iTrig].begin(), trackletsAccum[iTrig].end());
    triggerRecords[iTrig].setTrackletRange(tracklets.size() - nTracklets[iTrig], nTracklets[iTrig]);
  }

  auto processingTime = std::chrono::high_resolution_clock::now() - timeProcessingStart;

  LOG(info) << "Trap simulator found " << tracklets.size() << " tracklets from " << digits.size() << " Digits.";
  if (mUseMC) {
    LOG(info) << "In total " << lblTracklets.getNElements() << " MC labels are associated to the " << lblTracklets.getIndexedSize() << " tracklets";
  }
  LOG(info) << "Total processing time : " << std::chrono::duration_cast<std::chrono::milliseconds>(processingTime).count() << "ms";
  LOG(info) << "Digit Sorting took: " << std::chrono::duration_cast<std::chrono::milliseconds>(sortTime).count() << "ms";
  LOG(info) << "Processing time for parallel region: " << std::chrono::duration_cast<std::chrono::milliseconds>(parallelTime).count() << "ms";

  pc.outputs().snapshot(Output{"TRD", "TRACKLETS", 0, Lifetime::Timeframe}, tracklets);
  pc.outputs().snapshot(Output{"TRD", "TRKTRGRD", 0, Lifetime::Timeframe}, triggerRecords);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"TRD", "TRKLABELS", 0, Lifetime::Timeframe}, lblTracklets);
  }

  LOG(debug) << "TRD Trap Simulator Device exiting";
}

o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  inputs.emplace_back("digitinput", "TRD", "DIGITS", 0);
  inputs.emplace_back("triggerrecords", "TRD", "TRGRDIG", 0);

  outputs.emplace_back("TRD", "TRACKLETS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "TRKTRGRD", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labelinput", "TRD", "LABELS", 0);
    outputs.emplace_back("TRD", "TRKLABELS", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{"TRAP",
                           inputs,
                           outputs,
                           AlgorithmSpec{adaptFromTask<TRDDPLTrapSimulatorTask>(useMC)},
                           Options{
                             {"trd-trapconfig", VariantType::String, "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549", {"Name of the trap config from the CCDB default:cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549"}},
                             {"trd-onlinegaincorrection", VariantType::Bool, false, {"Apply online gain calibrations, mostly for back checking to run2 by setting FGBY to 0"}},
                             {"trd-onlinegaintable", VariantType::String, "Krypton_2015-02", {"Online gain table to be use, names found in CCDB, obviously trd-onlinegaincorrection must be set as well."}},
                             {"trd-dumptrapconfig", VariantType::Bool, false, {"Dump the selected trap configuration at loading time, to text file"}},
                             {"trd-runnum", VariantType::Int, 297595, {"Run number to use to anchor simulation to, defaults to 297595"}}}};
};

} //end namespace trd
} //end namespace o2
