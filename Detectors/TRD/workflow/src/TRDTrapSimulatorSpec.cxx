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

#include <chrono>
#include <optional>
#include <gsl/span>

#include "TFile.h"

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Lifetime.h"
#include "fairlogger/Logger.h"
#include "CCDB/BasicCCDBManager.h"

#include "TRDBase/Digit.h"
#include "TRDBase/Calibrations.h"
#include "DataFormatsTRD/TriggerRecord.h"
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

void TRDDPLTrapSimulatorTask::processTRAPchips(int currDetector, int& nTrackletsInTrigRec, std::vector<Tracklet64>& trapTrackletsAccum, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& lblTracklets, const o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>* lblDigits)
{
  // Loop over all TRAP chips of detector number currDetector.
  // TRAP chips without input data are skipped
  int currStack = (currDetector % NCHAMBERPERSEC) / NLAYER;
  int nTrapsMax = (currStack == 2) ? NROBC0 * NMCMROB : NROBC1 * NMCMROB;
  for (int iTrap = 0; iTrap < nTrapsMax; ++iTrap) {
    if (!mTrapSimulator[iTrap].isDataSet()) {
      continue;
    }
    auto timeTrapProcessingStart = std::chrono::high_resolution_clock::now();
    mTrapSimulator[iTrap].filter();
    mTrapSimulator[iTrap].tracklet();
    mTrapSimTime += std::chrono::high_resolution_clock::now() - timeTrapProcessingStart;
    auto trackletsOut = mTrapSimulator[iTrap].getTrackletArray64();
    int nTrackletsOut = trackletsOut.size();
    nTrackletsInTrigRec += nTrackletsOut;
    if (mUseMC) {
      auto digitCountOut = mTrapSimulator[iTrap].getTrackletDigitCount();     // number of digits contributing to each tracklet
      auto digitIndicesOut = mTrapSimulator[iTrap].getTrackletDigitIndices(); // global indices of the digits composing the tracklets
      int currDigitIndex = 0;                                                 // count the total number of digits which are associated to tracklets for this TRAP
      int trkltIdxStart = trapTrackletsAccum.size();
      for (int iTrklt = 0; iTrklt < nTrackletsOut; ++iTrklt) {
        // for each tracklet of this TRAP check the MC labels of the digits which contribute to the tracklet
        int tmp = currDigitIndex;
        for (int iDigitIndex = tmp; iDigitIndex < tmp + digitCountOut[iTrklt]; ++iDigitIndex) {
          if (iDigitIndex == tmp) {
            // for the first digit composing the tracklet we don't need to check for duplicate labels
            lblTracklets.addElements(trkltIdxStart + iTrklt, lblDigits->getLabels(digitIndicesOut[iDigitIndex]));
          } else {
            // in case more than one digit composes the tracklet we add only the labels
            // from the additional digit(s) which are not already contained in the previous
            // digit(s)
            auto currentLabels = lblTracklets.getLabels(trkltIdxStart + iTrklt);
            auto newLabels = lblDigits->getLabels(digitIndicesOut[iDigitIndex]);
            for (const auto& newLabel : newLabels) {
              bool isAlreadyIn = false;
              for (const auto& currLabel : currentLabels) {
                if (currLabel.compare(newLabel)) {
                  isAlreadyIn = true;
                }
              }
              if (!isAlreadyIn) {
                lblTracklets.addElement(trkltIdxStart + iTrklt, newLabel);
              }
            }
          }
          ++currDigitIndex;
        }
      }
    }
    trapTrackletsAccum.insert(trapTrackletsAccum.end(), trackletsOut.begin(), trackletsOut.end());
    mTrapSimulator[iTrap].reset();
  }
}

void TRDDPLTrapSimulatorTask::init(o2::framework::InitContext& ic)
{
  mShowTrackletStats = ic.options().get<int>("show-trd-trackletstats");
  mTrapConfigName = ic.options().get<std::string>("trd-trapconfig");
  mEnableOnlineGainCorrection = ic.options().get<bool>("trd-onlinegaincorrection");
  mOnlineGainTableName = ic.options().get<std::string>("trd-onlinegaintable");
  mShareDigitsManually = ic.options().get<bool>("trd-share-digits-manually");
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
  // this method steeres the processing of the TRAP simulation
  LOG(info) << "TRD Trap Simulator Device running over incoming message";

  // input
  auto inputDigits = pc.inputs().get<gsl::span<o2::trd::Digit>>("digitinput");                                 // block of TRD digits
  auto inputTriggerRecords = pc.inputs().get<std::vector<o2::trd::TriggerRecord>>("triggerrecords");           // time and number of digits for each collision
  // the above is changed to a vector from a span as the span is constant and cant modify elements.
  if (inputDigits.size() == 0 || inputTriggerRecords.size() == 0) {
    LOG(warn) << "Did not receive any digits, trigger records, or neither one nor the other. Aborting.";
    return;
  }
  LOG(debug) << "Read in " << inputDigits.size() << " digits";

  const o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>* lblDigitsPtr = nullptr;
  using lblType = std::decay_t<decltype(pc.inputs().get<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(""))>;
  std::optional<lblType> lblDigits;

  if (mUseMC) {
    lblDigits.emplace(pc.inputs().get<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>("labelinput")); // MC labels associated to the input digits
    lblDigitsPtr = &lblDigits.value();
    LOG(debug) << "Labels contain " << lblDigitsPtr->getNElements() << " elements with and indexed size of " << lblDigitsPtr->getIndexedSize();
    if (lblDigitsPtr->getIndexedSize() != inputDigits.size()) {
      LOG(warn) << "Digits and Labels coming into TrapSimulator are of differing sizes, labels will be jibberish. " << lblDigitsPtr->getIndexedSize() << "!=" << inputDigits.size();
    }
  }
  LOG(debug) << "Trigger records are available for " << inputTriggerRecords.size() << " collisions";

  // output
  std::vector<Tracklet64> trapTrackletsAccum;                          // calculated tracklets

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> lblTracklets;                                                    // MC labels for the tracklets, taken from the digits which make up the tracklet (duplicates are removed)

  // sort digits by chamber ID for each collision and keep track in index vector
  auto sortStart = std::chrono::high_resolution_clock::now();
  std::vector<unsigned int> digitIndices(inputDigits.size()); // digit indices sorted by chamber ID for each time frame
  std::iota(digitIndices.begin(), digitIndices.end(), 0);
  for (auto& trig : inputTriggerRecords) {
    std::stable_sort(std::begin(digitIndices) + trig.getFirstDigit(), std::begin(digitIndices) + trig.getNumberOfDigits() + trig.getFirstDigit(),
                     [&inputDigits](unsigned int i, unsigned int j) { return inputDigits[i].getDetector() < inputDigits[j].getDetector(); });
  }
  std::chrono::duration<double> sortTime = std::chrono::high_resolution_clock::now() - sortStart;

  // initialize timers
  auto timeDigitLoopStart = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> trapSimAccumulatedTime{};
  mTrapSimTime = std::chrono::duration<double>::zero();

  for (int iTrig = 0; iTrig < inputTriggerRecords.size(); ++iTrig) {
    int nTrackletsInTrigRec = 0;
    int currDetector = -1;
    for (int iDigit = inputTriggerRecords[iTrig].getFirstDigit(); iDigit < (inputTriggerRecords[iTrig].getFirstDigit() + inputTriggerRecords[iTrig].getNumberOfDigits()); ++iDigit) {
      const auto& digit = &inputDigits[digitIndices[iDigit]];
      if (currDetector < 0) {
        currDetector = digit->getDetector();
      }
      if (currDetector != digit->getDetector()) {
        // we switch to a new chamber, process all TRAPs of the previous chamber which contain data
        processTRAPchips(currDetector, nTrackletsInTrigRec, trapTrackletsAccum, lblTracklets, lblDigitsPtr);
        currDetector = digit->getDetector();
      }
      // fill the digit data into the corresponding TRAP chip
      int trapIdx = digit->getROB() * NMCMROB + digit->getMCM();
      if (!mTrapSimulator[trapIdx].isDataSet()) {
        mTrapSimulator[trapIdx].init(mTrapConfig, digit->getDetector(), digit->getROB(), digit->getMCM());
      }
      if (digit->isSharedDigit() && mShareDigitsManually) {
        LOG(error) << "Digit duplication requested, but found shared digit in input stream. Digits will be duplicated twice.";
      }
      if (mShareDigitsManually) {
        if ((digit->getChannel() == 2) && !((digit->getROB() % 2 != 0) && (digit->getMCM() % NMCMROBINCOL == 3))) {
          // shared left, if not leftmost MCM of left ROB of chamber
          int robShared = (digit->getMCM() % NMCMROBINCOL == 3) ? digit->getROB() + 1 : digit->getROB(); // for the leftmost MCM on a ROB the shared digit is added to the neighbouring ROB
          int mcmShared = (robShared == digit->getROB()) ? digit->getMCM() + 1 : digit->getMCM() - 3;
          int trapIdxLeft = robShared * NMCMROB + mcmShared;
          if (!mTrapSimulator[trapIdxLeft].isDataSet()) {
            mTrapSimulator[trapIdxLeft].init(mTrapConfig, digit->getDetector(), robShared, mcmShared);
          }
          mTrapSimulator[trapIdxLeft].setData(NADCMCM - 1, digit->getADC(), digitIndices[iDigit]);
        }
        if ((digit->getChannel() == 18 || digit->getChannel() == 19) && !((digit->getROB() % 2 == 0) && (digit->getMCM() % NMCMROBINCOL == 0))) {
          // shared right, if not rightmost MCM of right ROB of chamber
          int robShared = (digit->getMCM() % NMCMROBINCOL == 0) ? digit->getROB() - 1 : digit->getROB(); // for the rightmost MCM on a ROB the shared digit is added to the neighbouring ROB
          int mcmShared = (robShared == digit->getROB()) ? digit->getMCM() - 1 : digit->getMCM() + 3;
          int trapIdxRight = robShared * NMCMROB + mcmShared;
          if (!mTrapSimulator[trapIdxRight].isDataSet()) {
            mTrapSimulator[trapIdxRight].init(mTrapConfig, digit->getDetector(), robShared, mcmShared);
          }
          mTrapSimulator[trapIdxRight].setData(digit->getChannel() - NCOLMCM, digit->getADC(), digitIndices[iDigit]);
        }
      }
      mTrapSimulator[trapIdx].setData(digit->getChannel(), digit->getADC(), digitIndices[iDigit]);
    }
    // take care of the TRAPs for the last chamber
    processTRAPchips(currDetector, nTrackletsInTrigRec, trapTrackletsAccum, lblTracklets, lblDigitsPtr);
    inputTriggerRecords[iTrig].setTrackletRange(trapTrackletsAccum.size() - nTrackletsInTrigRec, nTrackletsInTrigRec);
  }

  LOG(info) << "Trap simulator found " << trapTrackletsAccum.size() << " tracklets from " << inputDigits.size() << " Digits.";
  if (mUseMC) {
    LOG(info) << "In total " << lblTracklets.getNElements() << " MC labels are associated to the tracklets";
  }
  if (mShowTrackletStats > 0) {
    std::chrono::duration<double> digitLoopTime = std::chrono::high_resolution_clock::now() - timeDigitLoopStart;
    LOG(info) << "Trap Simulator done ";
    LOG(info) << "Digit Sorting took " << sortTime.count() << "s";
    LOG(info) << "Digit loop took : " << digitLoopTime.count() << "s";
    LOG(info) << "TRAP processing TrapSimulator::filter() + TrapSimulator::tracklet() took : " << trapSimAccumulatedTime.count() << "s";
  }
  pc.outputs().snapshot(Output{"TRD", "TRACKLETS", 0, Lifetime::Timeframe}, trapTrackletsAccum);
  pc.outputs().snapshot(Output{"TRD", "TRKTRGRD", 0, Lifetime::Timeframe}, inputTriggerRecords);
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
                             {"show-trd-trackletstats", VariantType::Int, 1, {"Display the processing time of the tracklet processing in the TRAPs"}},
                             {"trd-trapconfig", VariantType::String, "cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549", {"Name of the trap config from the CCDB default:cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549"}},
                             {"trd-onlinegaincorrection", VariantType::Bool, false, {"Apply online gain calibrations, mostly for back checking to run2 by setting FGBY to 0"}},
                             {"trd-onlinegaintable", VariantType::String, "Krypton_2015-02", {"Online gain table to be use, names found in CCDB, obviously trd-onlinegaincorrection must be set as well."}},
                             {"trd-dumptrapconfig", VariantType::Bool, false, {"Dump the selected trap configuration at loading time, to text file"}},
                             {"trd-share-digits-manually", VariantType::Bool, false, {"Duplicate digits connected to shared pads if the digitizer did not already do so."}},
                             {"trd-runnum", VariantType::Int, 297595, {"Run number to use to anchor simulation to, defaults to 297595"}}}};
};

} //end namespace trd
} //end namespace o2
