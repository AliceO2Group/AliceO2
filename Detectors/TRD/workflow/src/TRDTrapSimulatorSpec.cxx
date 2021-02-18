// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//#include "Framework/runDataProcessing.h"
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
#include "TStopwatch.h"
#include "Steer/HitProcessingManager.h" // for DigitizationContext
#include "TChain.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include "fairlogger/Logger.h"
#include "CCDB/BasicCCDBManager.h"

#include "DataFormatsParameters/GRPObject.h"
#include "TRDSimulation/TrapSimulator.h"
#include "TRDSimulation/Digitizer.h"
#include "TRDSimulation/Detector.h"
#include "TRDBase/Digit.h" // for the Digit type
#include "TRDBase/FeeParam.h"
#include "TRDBase/Calibrations.h"
#include "TRDSimulation/TrapSimulator.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/LinkRecord.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/RawData.h"
#include "DataFormatsTRD/Constants.h"

using namespace o2::framework;
using namespace std::placeholders; // this is for std::bind to build the comparator for the indexed sort of digits.

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
  mFeeParam = FeeParam::instance();
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

bool digitindexcompare(unsigned int A, unsigned int B, const std::vector<o2::trd::Digit>& originalDigits)
{
  // sort into ROC:padrow
  const o2::trd::Digit *a, *b;
  a = &originalDigits[A];
  b = &originalDigits[B];
  if (a->getDetector() < b->getDetector()) {
    return 1;
  }
  if (a->getDetector() > b->getDetector()) {
    return 0;
  }
  if (a->getRow() < b->getRow()) {
    return 1;
  }
  if (a->getRow() > b->getRow()) {
    return 0;
  }
  return 0;
}

void TRDDPLTrapSimulatorTask::setTriggerRecord(std::vector<o2::trd::TriggerRecord>& triggerrecord, uint32_t currentrecord, uint64_t recordsize)
{
  // so increment the tracklet trigger records and fill accordingly for the now completed prior triggerrecord.
  uint64_t triggerrecordstart = 0;
  if (currentrecord == 0) { // for not the first one we can simply look back to the previous one to get the start.
    triggerrecordstart = 0;
    triggerrecord[currentrecord].setDataRange(triggerrecordstart, recordsize);
  } else {
    triggerrecordstart = triggerrecord[currentrecord - 1].getFirstEntry() + triggerrecord[currentrecord - 1].getNumberOfObjects();
    triggerrecord[currentrecord].setDataRange(triggerrecordstart, recordsize - triggerrecordstart);
  }
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
  std::vector<o2::trd::Digit> msgDigits(inputDigits.begin(), inputDigits.end()); // TODO: check if sorting the digits is still required. If not, this copy can be removed
  auto digitMCLabels = pc.inputs().get<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>("labelinput");
  auto inputTriggerRecords = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("triggerrecords");

  if (msgDigits.size() == 0 || inputTriggerRecords.size() == 0) {
    LOG(WARNING) << "Did not receive any digits, trigger records, or neither one nor the other. Aborting.";
    return;
  }

  /* *****
   * setup data objects
   * *****/

  // trigger records to index the 64bit tracklets.yy
  std::vector<o2::trd::TriggerRecord> trackletTriggerRecords(inputTriggerRecords.begin(), inputTriggerRecords.end()); // copy over the whole thing but we only really want the bunch crossing info.
  std::vector<o2::trd::TriggerRecord> rawTriggerRecords = trackletTriggerRecords;                                     // as we have the option of having tracklets and/or raw data, we need both triggerrecords.
                                                                                                                      // of course we dont *actually* need it we could simply walk through all the raw data header to header.
  //  auto rawDataOut = pc.outputs().make<char>(Output{"TRD", "RAWDATA", 0, Lifetime::Timeframe}, 1000); //TODO number is just a place holder until we start using it.
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> trackletMCLabels;
  //index of digits, TODO refactor to a digitindex class.
  std::vector<unsigned int> msgDigitsIndex(msgDigits.size());
  //set up structures to hold the returning tracklets.
  std::vector<Tracklet64> trapTracklets; //vector to store the retrieved tracklets from an trapsim object
  std::vector<Tracklet64> trapTrackletsAccum;
  std::vector<uint32_t> rawdata;
  // trigger records to index the "raw" data
  uint64_t currentTriggerRecord = 0;

  /* *******
   * reserve sizes
   * *******/
  mLinkRecords.reserve(2 * MAXCHAMBER * inputTriggerRecords.size()); // worse case scenario is all links for all events
  LOG(debug) << "Read in msgDigits with size of : " << msgDigits.size() << " labels contain : " << digitMCLabels.getNElements() << " with and index size of  : " << digitMCLabels.getIndexedSize() << " and triggerrecord count of :" << inputTriggerRecords.size();
  if (digitMCLabels.getIndexedSize() != msgDigits.size()) {
    LOG(warn) << "Digits and Labels coming into TrapSimulator are of differing sizes, labels will be jibberish. " << digitMCLabels.getIndexedSize() << "!=" << msgDigits.size();
  }
  trapTracklets.reserve(30);
  trapTrackletsAccum.reserve(msgDigits.size() / 3);
  //msgDigitsIndex.reserve(msgDigits.size());
  // worse case scenario is header and single tracklet word, hence 2, for higher tracklet count the factors reduces relative to tracklet count. Remember 3 digits per tracklet.
  rawdata.reserve(msgDigits.size() * 2);

  //Build the digits index.
  std::iota(msgDigitsIndex.begin(), msgDigitsIndex.end(), 0);
  //sort the digits array TODO refactor this intoa vector index sort and possibly generalise past merely digits.
  auto sortstart = std::chrono::high_resolution_clock::now();
  // TODO check if sorting is still needed
  for (auto& trig : inputTriggerRecords) {
    std::stable_sort(std::begin(msgDigitsIndex) + trig.getFirstEntry(), std::begin(msgDigitsIndex) + trig.getNumberOfObjects() + trig.getFirstEntry(),
                     [&msgDigits](auto&& PH1, auto&& PH2) { return digitindexcompare(PH1, PH2, msgDigits); });
  }

  mSortingTime = std::chrono::high_resolution_clock::now() - sortstart;
  LOG(debug) << "TRD Digit Sorting took " << mSortingTime.count();
  //print digits to check the sorting.
  LOG(debug) << " Digits : ";
  //for (auto& digit : msgDigits) {
  for (auto& digitindex : msgDigitsIndex) {
    Digit digit = msgDigits[digitindex];
    LOG(debug) << "sorted digit detector:row:pad:rob:mcm ::"
               << digit.getDetector() << ":" << digit.getRow() << ":" << digit.getPad() << ":"
               << mFeeParam->getROBfromPad(digit.getRow(), digit.getPad()) << ":"
               << mFeeParam->getMCMfromPad(digit.getRow(), digit.getPad())
               << " LinkId:" << LinkRecord::getHalfChamberLinkId(digit.getDetector(), mFeeParam->getROBfromPad(digit.getRow(), digit.getPad())) << "\t\t  SM:stack:layer:side  "
               << digit.getDetector() / 30 << ":" << Geometry::getStack(digit.getDetector())
               << ":" << Geometry::getLayer(digit.getDetector()) << ":" << FeeParam::instance()->getRobSide(mFeeParam->getROBfromPad(digit.getRow(), digit.getPad()))
               << " with ORI# : " << mFeeParam->getORI(digit.getDetector(), mFeeParam->getROBfromPad(digit.getRow(), digit.getPad()))
               << " within SM ori#:" << mFeeParam->getORIinSM(digit.getDetector(), mFeeParam->getROBfromPad(digit.getRow(), digit.getPad()));
  }

  //accounting variables for various things.
  int loopindex = 0;
  int counttrackletadditions = 0;
  int oldsize = 0;
  unsigned long oldtrackletcount = 0;
  mTotalRawWordsWritten = 0; // words written for the raw format of 4x32bits, where 4 can be 2 to 4 depending on # of tracklets in the block.
  mOldHalfChamberLinkId = 0;
  mNewTrackletHCHeaderHasBeenWritten = false;
  int countProcessedTraps = 0;

  // now to loop over the incoming digits.
  auto digitloopstart = std::chrono::high_resolution_clock::now();
  uint64_t digitcounter = 0;
  // set current detector and row for the first digit
  int currDetector = msgDigits[msgDigitsIndex[0]].getDetector();
  int currRow = msgDigits[msgDigitsIndex[0]].getRow();
  mOldHalfChamberLinkId = LinkRecord::getHalfChamberLinkId(msgDigits[msgDigitsIndex[0]].getDetector(), msgDigits[msgDigitsIndex[0]].getROB());
  for (const auto digitIdx : msgDigitsIndex) {
    // while on a single padrow, populate data structures in the 8 trap simulators
    // on change of padrow fire up trap simulators, do its thing with the data which has been written for the previous padrow
    auto digit = &msgDigits[digitIdx];

    //Are we on a new half chamber ?
    if (mOldHalfChamberLinkId != LinkRecord::getHalfChamberLinkId(digit->getDetector(), digit->getROB())) {
      // OS: I do not understand this block. What do we need it for? digit -> raw conversion?
      // If we don't sort the digits column-wise we might switch back and forth between half chambers -> so maybe sort digits by pad column again if this block is needed?
      //     hcid= detector*2 + robpos%2;
      // new half chamber so add the header to the raw data stream.
      buildTrackletHCHeaderd(mTrackletHCHeader, digit->getDetector(), digit->getROB(), currentTriggerRecord * 42, 4);
      //buildTrackletHCHeader(mTrackletHCHeader,sector,stack,layer,side,currentTriggerRecord*42,4);
      mOldHalfChamberLinkId = LinkRecord::getHalfChamberLinkId(digit->getDetector(), digit->getROB());
      // now we have a problem. We must only write the halfchamberheader if a tracklet is written i.e. if the digits for this half chamber actually produce 1 or more tracklets!
      mNewTrackletHCHeaderHasBeenWritten = false;
    }
    //figure out which trigger record from digits we are on
    if (digitcounter >= inputTriggerRecords[currentTriggerRecord].getFirstEntry() + inputTriggerRecords[currentTriggerRecord].getNumberOfObjects()) {
      //trigger record changed.
      //Now we know the ranges so populate the triggerrecord related to the previously block of data.
      // what if the trigger record changes, but we only simulate one chamber (particle gun)..
      // then all data also from different collisions is fed into the MCMs and processed at the same time -> ERROR -> need to initiate TRAP processing for already loaded data on change of trigger record
      setTriggerRecord(trackletTriggerRecords, currentTriggerRecord, trapTrackletsAccum.size());
      setTriggerRecord(rawTriggerRecords, currentTriggerRecord, mTotalRawWordsWritten);
      currentTriggerRecord++;
      LOG(debug) << "changing trigger records : " << currentTriggerRecord;
    }

    if (currDetector != digit->getDetector() || currRow != digit->getRow()) {
      // we have gone over the pad row. //TODO ??? do we need to check for change of time as well?
      // -> change of time would mean new trigger record. But in case of pile up...? Basically in real data taking pile up cannot be separated from the signal, so I think it is not needed to check the time
      // the labels might tell in the end if the digit is from pile up or not..
      //all data is inside the 8 relavent trapsimulators
      //fireup Trapsim.
      auto traploopstart = std::chrono::high_resolution_clock::now();
      unsigned long numberofusedtraps = 0;
      for (int trapcounter = 0; trapcounter < 8; trapcounter++) {
        if (!mTrapSimulator[trapcounter].checkInitialized()) {
          LOG(debug) << "TRAP chip not initialized, as there is no input data set. Skipping it.";
          continue;
        }
        if (!mTrapSimulator[trapcounter].isDataSet()) {
          LOG(debug) << "TRAP chip initialized, but no data for this pad row. Skipping it.";
          continue;
        }
        countProcessedTraps++;
        //this one has been filled with data for the now previous pad row.
        auto trapsimtimerstart = std::chrono::high_resolution_clock::now();
        mTrapUsedCounter[trapcounter]++;
        numberofusedtraps++;
        mTrapSimulator[trapcounter].filter();
        mTrapSimulator[trapcounter].tracklet();

        trapTracklets = mTrapSimulator[trapcounter].getTrackletArray64(); //TODO remove the copy and send the Accumulated array into the Trapsimulator
        auto trapLabels = mTrapSimulator[trapcounter].getTrackletLabels();
        if (!mNewTrackletHCHeaderHasBeenWritten && trapTracklets.size() != 0) { // take account of the case where we have data in the trapchip adc but no tracklets // OS: Is this a TODO??
          //we have a tracklet for said half chamber, but the halfchamber ID has not been written yet
          // .. fix the previous linkrecord to note its end of range.
          if (mLinkRecords.size() == 0) { // special case for the first entry into the linkrecords vector.
            mLinkRecords.emplace_back(mTrackletHCHeader.word, 0, -1);
            //   LOG(debug) << " added HCID :[record.size==0] " << mTrackletHCHeader.HCID << " with number of bytes : " << mTotalRawWordsWritten << "-" << mLinkRecords.back().getFirstEntry();
          } else {
            mLinkRecords.back().setNumberOfObjects(mTotalRawWordsWritten - mLinkRecords.back().getFirstEntry()); // current number of words written - the start of this index record.
                                                                                                                 //  LOG(debug) << " added HCID : " << mTrackletHCHeader.HCID << " with number of bytes : " << mTotalRawWordsWritten << "-" << mLinkRecords.back().getFirstEntry();
            //..... so write the new one thing
            mLinkRecords.emplace_back(mTrackletHCHeader.word, mTotalRawWordsWritten, -1); // set the number of elements to -1 for an error condition
          }
          mNewTrackletHCHeaderHasBeenWritten = true;
          LOG(debug) << mTrackletHCHeader;
        }
        LOG(debug) << "getting trackletsteram for trapcounter = " << trapcounter;
        auto wordswritten = mTrapSimulator[trapcounter].getTrackletStream(rawdata, mTotalRawWordsWritten); // view of data from current marker and only 5 words long (can only have 4 words at most in the trackletstream for 1 MCM)
        mTotalRawWordsWritten += wordswritten;
        LOG(debug) << "Tracklets accumulated before addition of new ones :" << trapTrackletsAccum.size()
                   << " :: about to add " << trapTracklets.size() << " count tracklets, labels coming in index of: "
                   << trapLabels.getIndexedSize() << " and elements of : "
                   << trapLabels.getNElements() << " with countrackletadditions :  " << counttrackletadditions;
        counttrackletadditions++;
        trapTrackletsAccum.insert(trapTrackletsAccum.end(), trapTracklets.begin(), trapTracklets.end());
        trackletMCLabels.mergeAtBack(trapLabels);
        mTrapSimAccumulatedTime += std::chrono::high_resolution_clock::now() - trapsimtimerstart;
        if (mShowTrackletStats > 0) {
          if (trapTrackletsAccum.size() - oldsize > mShowTrackletStats) {
            LOG(debug) << "TrapSim Accumulated tracklets: " << trapTrackletsAccum.size() << " :: " << trapTracklets.size();
            oldsize = trapTrackletsAccum.size();
          }
        }

        // mTrapSimulator[trapcounter].zeroSupressionMapping();

        if (mDrawTrackletOptions != 0) {
          mTrapSimulator[trapcounter].draw(mDrawTrackletOptions, loopindex);
        }
        if (mDebugRejectedTracklets) {                    //&& trapTracklets.size()==0) {
          mTrapSimulator[trapcounter].draw(7, loopindex); //draw adc when no tracklets are found.A
          LOG(debug) << "loop index  : " << loopindex;
          //mTrapSimulator[trapcounter].print(1);
        }
        if (mPrintTrackletOptions != 0) {
          mTrapSimulator[trapcounter].print(mPrintTrackletOptions);
        }

        loopindex++;
        // reset trap sim object
        mTrapSimulator[trapcounter].reset();
      }
      //timing info
      mTrapLoopTime += std::chrono::high_resolution_clock::now() - traploopstart;
      mTrapUsedFrequency[numberofusedtraps]++;

      LOG(debug) << "Row change ... Tracklets so far: " << trapTrackletsAccum.size();
      if (mShowTrackletStats > 0) {
        if (trapTrackletsAccum.size() - oldtrackletcount > mShowTrackletStats) {
          oldtrackletcount = trapTrackletsAccum.size();
          unsigned long mcmTrackletsize = trapTrackletsAccum.size();
          mTrackletTime = std::chrono::high_resolution_clock::now() - digitloopstart;
          double trackletrate = mcmTrackletsize / mTrackletTime.count();
          LOG(info) << "Getting tracklets at the rate of : " << trackletrate << " Tracklets/s ... Accumulated tracklets : " << trapTrackletsAccum.size();
        }
      }
      currDetector = digit->getDetector();
      currRow = digit->getRow();
    } // end processing of TRAP chips in given padrow

    //we are still on the same detector and row.
    //add the digits to the padrow.
    //copy pad time data into where they belong in the 8 TrapSimulators for this padrow.
    int trapindex = (digit->getMCM() % NMCMROBINCOL) + (digit->getROB() % 2) * NMCMROBINCOL; // index of TRAP chip within padrow [0..7]
    //check trap is initialised.
    if (!mTrapSimulator[trapindex].isDataSet()) {
      LOG(debug) << "Initialising trapsimulator for det:rob:mcm " << digit->getDetector() << ":" << digit->getROB() << ":"
                 << digit->getMCM() << " as this is the first ADC data to be set for this mcm";
      mTrapSimulator[trapindex].init(mTrapConfig, digit->getDetector(), digit->getROB(), digit->getMCM());
    }
    std::vector<o2::MCCompLabel> tmplabels;
    auto digitslabels = digitMCLabels.getLabels(digitcounter);
    for (auto& tmplabel : digitslabels) {
      tmplabels.push_back(tmplabel);
    }
    LOG(debug) << "tmplabels for set data : " << tmplabels.size() << " and gslspan digitlabels size of : " << digitslabels.size();
    LOG(debug) << " setting data with pad=" << digit->getPad() << " ti=" << trapindex + 1;
    mTrapSimulator[trapindex].setData(digit->getChannel(), digit->getADC(), tmplabels);

    digitcounter++;
  } // end of loop over digits.
  // don't forget about the last pad row..
  for (int trapcounter = 0; trapcounter < 8; trapcounter++) {
    if (!mTrapSimulator[trapcounter].checkInitialized()) {
      LOG(debug) << "TRAP chip not initialized, as there is no input data set. Skipping it.";
      continue;
    }
    if (!mTrapSimulator[trapcounter].isDataSet()) {
      LOG(debug) << "TRAP chip initialized, but no data for this pad row. Skipping it.";
      continue;
    }
    countProcessedTraps++;
    //this one has been filled with data for the last pad row.
    auto trapsimtimerstart = std::chrono::high_resolution_clock::now();
    mTrapUsedCounter[trapcounter]++;
    mTrapSimulator[trapcounter].filter();
    mTrapSimulator[trapcounter].tracklet();

    trapTracklets = mTrapSimulator[trapcounter].getTrackletArray64(); //TODO remove the copy and send the Accumulated array into the Trapsimulator
    auto trapLabels = mTrapSimulator[trapcounter].getTrackletLabels();
    if (!mNewTrackletHCHeaderHasBeenWritten && trapTracklets.size() != 0) { // take account of the case where we have data in the trapchip adc but no tracklets OS: Is this a TODO?
      //we have a tracklet for said half chamber, but the halfchamber ID has not been written yet
      // .. fix the previous linkrecord to note its end of range.
      if (mLinkRecords.size() == 0) { // special case for the first entry into the linkrecords vector.
        mLinkRecords.emplace_back(mTrackletHCHeader.word, 0, -1);
        //   LOG(debug) << " added HCID :[record.size==0] " << mTrackletHCHeader.HCID << " with number of bytes : " << mTotalRawWordsWritten << "-" << mLinkRecords.back().getFirstEntry();
      } else {
        mLinkRecords.back().setNumberOfObjects(mTotalRawWordsWritten - mLinkRecords.back().getFirstEntry()); // current number of words written - the start of this index record.
                                                                                                             //  LOG(debug) << " added HCID : " << mTrackletHCHeader.HCID << " with number of bytes : " << mTotalRawWordsWritten << "-" << mLinkRecords.back().getFirstEntry();
        //..... so write the new one thing
        mLinkRecords.emplace_back(mTrackletHCHeader.word, mTotalRawWordsWritten, -1); // set the number of elements to -1 for an error condition
      }
      mNewTrackletHCHeaderHasBeenWritten = true;
      LOG(debug) << mTrackletHCHeader;
    }

    auto wordswritten = mTrapSimulator[trapcounter].getTrackletStream(rawdata, mTotalRawWordsWritten); // view of data from current marker and only 5 words long (can only have 4 words at most in the trackletstream for 1 MCM)
    mTotalRawWordsWritten += wordswritten;

    trapTrackletsAccum.insert(trapTrackletsAccum.end(), trapTracklets.begin(), trapTracklets.end());
    trackletMCLabels.mergeAtBack(trapLabels);
    mTrapSimAccumulatedTime += std::chrono::high_resolution_clock::now() - trapsimtimerstart;

    // reset trap sim object
    mTrapSimulator[trapcounter].reset();
  } // done with last padrow

  // now finalise
  auto triggerrecordstart = trackletTriggerRecords[currentTriggerRecord - 1].getFirstEntry() + trackletTriggerRecords[currentTriggerRecord - 1].getNumberOfObjects();
  trackletTriggerRecords[currentTriggerRecord].setDataRange(triggerrecordstart, trapTrackletsAccum.size() - triggerrecordstart);
  mLinkRecords.back().setNumberOfObjects(mTotalRawWordsWritten - mLinkRecords.back().getFirstEntry()); // set the final link entry
  LOG(info) << "Trap simulator found " << trapTrackletsAccum.size() << " tracklets from " << msgDigits.size() << " Digits and " << trackletMCLabels.getIndexedSize() << " associated MC Label indexes and " << trackletMCLabels.getNElements() << " associated MC Labels";
  if (mShowTrackletStats > 0) {
    mDigitLoopTime = std::chrono::high_resolution_clock::now() - digitloopstart;
    LOG(info) << "Trap Simulator done ";
#ifdef TRDTIMESORT
    LOG(info) << "Sorting took " << mSortingTime.count() << "s";
#endif
    LOG(info) << "Digit loop took : " << mDigitLoopTime.count() << "s";
    LOG(info) << "Trapsim took : " << mTrapSimAccumulatedTime.count() << "s";
    LOG(info) << "Traploop took : " << mTrapLoopTime.count() << "s";
    for (auto trapcount : mTrapUsedFrequency) {
      LOG(info) << "# traps fired Traploop are : " << trapcount;
    }
    for (auto trapcount : mTrapUsedCounter) {
      LOG(info) << "each trap position fired   : " << trapcount;
    }
    LOG(info) << "Raw data words written = " << mTotalRawWordsWritten << " with a vector size = " << rawdata.size();
    LOG(info) << "Raw data words written = " << mTotalRawWordsWritten << " with a vector size = " << rawdata.size();
  }
  LOG(debug) << "END OF RUN .............";
  //TODO does anyone care to have the digits to tracklet mapping. Do we then presort the digits inline with the index or send both digits and sorted index.
  //TODO is is available for post processing via the debug stream output.
  pc.outputs().snapshot(Output{"TRD", "TRACKLETS", 0, Lifetime::Timeframe}, trapTrackletsAccum);
  pc.outputs().snapshot(Output{"TRD", "TRKTRGRD", 0, Lifetime::Timeframe}, trackletTriggerRecords);
  /*pc.outputs().snapshot(Output{"TRD", "TRKLABELS", 0, Lifetime::Timeframe}, trackletMCLabels);  */
  // LOG(info) << "digit MCLabels is of type : " << type_id_with_cvr<decltype(digitMCLabels)>().pretty_name();
  pc.outputs().snapshot(Output{"TRD", "RAWDATA", 0, Lifetime::Timeframe}, rawdata);
  pc.outputs().snapshot(Output{"TRD", "RAWTRGRD", 0, Lifetime::Timeframe}, rawTriggerRecords);
  pc.outputs().snapshot(Output{"TRD", "RAWLNKRD", 0, Lifetime::Timeframe}, mLinkRecords);

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
                                   OutputSpec{"TRD", "TRKLABELS", 0, Lifetime::Timeframe},
                                   /*OutputSpec{"TRD", "TRAPRAWDUMP", 0, Lifetime::Timeframe},*/
                                   OutputSpec{"TRD", "RAWTRGRD", 0, Lifetime::Timeframe}, // offsets for each event in the rawdata
                                   OutputSpec{"TRD", "RAWLNKRD", 0, Lifetime::Timeframe}, // offsets for each link/halfchamberheader in the rawdata, halfchamberheader sitting in here.
                                   OutputSpec{"TRD", "RAWDATA", 0, Lifetime::Timeframe}}, // this is the mcmheader,traprawtracklet, repeat in varying quantities.
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
