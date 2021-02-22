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

//#ifdef WITH_OPENMP
//#include <omp.h>
//#endif ci is failing on this, sort out another time.

using namespace o2::framework;
using namespace std::placeholders; // this is for std::bind to build the comparator for the indexed sort of digits.

namespace o2
{
namespace trd
{

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
  const int nDets = 540; //Geometry::Ndet();
  const int nMcms = Geometry::MCMmax();
  const int nChs = Geometry::ADCmax();
  //check FGBY from trapconfig.
  //check the input parameter of trd-onlinegaincorrection.
  //warn if you have chosen a trapconfig with gaincorrections but chosen not to use them.
  if (mEnableOnlineGainCorrection) {
    if (mTrapConfig->getTrapReg(TrapConfig::kFGBY) == 0) {
      LOG(warn) << "you have asked to do online gain calibrations but the selected trap config does not have FGBY enabled, so modifying trapconfig to conform to your command line request. OnlineGains will be 1, i.e. no effect.";
      for (int iDet = 0; iDet < nDets; ++iDet) {
        mTrapConfig->setTrapReg(TrapConfig::kFGBY, 1, iDet);
      }
    }
    mCalib->setOnlineGainTables(mOnlineGainTableName);
    //TODO add some error checking inhere.
    // gain factors are per MCM
    // allocate the registers accordingly
    for (int ch = 0; ch < nChs; ++ch) {
      TrapConfig::TrapReg_t regFGAN = (TrapConfig::TrapReg_t)(TrapConfig::kFGA0 + ch);
      TrapConfig::TrapReg_t regFGFN = (TrapConfig::TrapReg_t)(TrapConfig::kFGF0 + ch);
      mTrapConfig->setTrapRegAlloc(regFGAN, TrapConfig::kAllocByMCM);
      mTrapConfig->setTrapRegAlloc(regFGFN, TrapConfig::kAllocByMCM);
    }

    for (int iDet = 0; iDet < nDets; ++iDet) {
      const int nRobs = Geometry::getStack(iDet) == 2 ? Geometry::ROBmaxC0() : Geometry::ROBmaxC1();
      for (int rob = 0; rob < nRobs; ++rob) {
        for (int mcm = 0; mcm < nMcms; ++mcm) {
          // set ADC reference voltage
          mTrapConfig->setTrapReg(TrapConfig::kADCDAC, mCalib->getOnlineGainAdcdac(iDet, rob, mcm), iDet, rob, mcm);
          // set constants channel-wise
          for (int ch = 0; ch < nChs; ++ch) {
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

std::string printDigit(o2::trd::Digit& a)
{
  std::string out;
  out = a.getRow() + std::string(".") + a.getPad(); // + std::string(".")+ a.getLabelIndex();
  return out;
}

bool digitindexcompare(unsigned int A, unsigned int B, const std::vector<o2::trd::Digit>& originalDigits)
{
  // sort into ROC:padrow:padcolum
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
  if (a->getPad() < b->getPad()) {
    return 0;
  }
  if (a->getPad() > b->getPad()) {
    return 1;
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

void TRDDPLTrapSimulatorTask::fixTriggerRecords(std::vector<o2::trd::TriggerRecord>& trigRecord)
{
  // Trigger records are coming with an extra one at the end, and the first one blank and the last 2 having the same bunch crossing information.
  // This is temporary.

  // take the nth records DataRange, and insert it into the (n-1)th DataRange
  // thereby realigning the bunch crossings and dataranges.
  // drop the final entry.

  //sanity check -- this is only true if the first range is 0 to 0
  if (trigRecord[0].getFirstEntry() == 0 && trigRecord[0].getNumberOfObjects() == 0) {
    for (int i = 0; i < trigRecord.size() - 1; i++) {
      trigRecord[i].setDataRange(trigRecord[i + 1].getFirstEntry(), trigRecord[i + 1].getNumberOfObjects());
    }
    //now drop the final triggerrecord.
    trigRecord.pop_back();
  } else {
    LOG(warn) << "TriggerRecord fix requested, but inital TriggerRecord is not 0,0";
  }
}

void TRDDPLTrapSimulatorTask::run(o2::framework::ProcessingContext& pc)
{
  LOG(info) << "TRD Trap Simulator Device running over incoming message";

  //#ifdef WITH_OPENMP
  //  int maxthreads = omp_get_max_threads();
  //  mNumThreads = std::min(maxthreads, 8);
  //  LOG(INFO) << "TRD: Trapping with " << mNumThreads << " threads ";
  //#endif

  // get inputs for the TrapSimulator
  // the digits are going to be sorted, we therefore need a copy of the vector rather than an object created
  // directly on the input data, the output vector however is created directly inside the message
  // memory thus avoiding copy by snapshot

  /*********
   * iNPUTS
   ********/

  auto inputDigits = pc.inputs().get<gsl::span<o2::trd::Digit>>("digitinput");
  std::vector<o2::trd::Digit> msgDigits(inputDigits.begin(), inputDigits.end());
  //  auto digits pc.outputs().make<std::vector<o2::trd::Digit>>(Output{"TRD", "TRKDIGITS", 0, Lifetime::Timeframe}, msgDigits.begin(), msgDigits.end());
  auto digitMCLabels = pc.inputs().get<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>("labelinput");
  // the returned object is read-only as it refers directly to the underlying raw input data
  // need to make a copy because the object might be changed in fixTriggerRecords
  auto inputTriggerRecords = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("triggerrecords");

  /* *****
   * setup data objects
   * *****/

  // trigger records to index the 64bit tracklets.yy
  std::vector<o2::trd::TriggerRecord> triggerRecords(inputTriggerRecords.begin(), inputTriggerRecords.end());
  std::vector<o2::trd::TriggerRecord> trackletTriggerRecords = triggerRecords; // copy over the whole thing but we only really want the bunch crossing info.
  std::vector<o2::trd::TriggerRecord> rawTriggerRecords = triggerRecords;      // as we have the option of having tracklets and/or raw data, we need both triggerrecords.
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
  mLinkRecords.reserve(1080 * triggerRecords.size()); // worse case scenario is all links for all events. TODO get 1080 from somewhere.
  //TODO these must be created directly in the output as done at the top of this run method
  //msgDigitsIndex.reserve(msgDigits.size());
  LOG(debug) << "Read in msgDigits with size of : " << msgDigits.size() << " labels contain : " << digitMCLabels.getNElements() << " with and index size of  : " << digitMCLabels.getIndexedSize() << " and triggerrecord count of :" << triggerRecords.size();
  if (digitMCLabels.getIndexedSize() != msgDigits.size()) {
    LOG(warn) << "Digits and Labels coming into TrapSimulator are of differing sizes, labels will be jibberish. " << digitMCLabels.getIndexedSize() << "!=" << msgDigits.size();
  }
  trapTracklets.reserve(30);
  trapTrackletsAccum.reserve(msgDigits.size() / 3);
  //msgDigitsIndex.reserve(msgDigits.size());
  // worse case scenario is header and single tracklet word, hence 2, for higher tracklet count the factors reduces relative to tracklet count. Remember 3 digits per tracklet.
  rawdata.reserve(msgDigits.size() * 2);

  //Build the digits index.
  //  std::iota(msgDigitsIndex.begin(), msgDigitsIndex.end(), static_cast<unsigned int>(0));
  std::generate(msgDigitsIndex.begin(), msgDigitsIndex.end(), [n = 0]() mutable { return n++; });
  int indexcount = 0;
  for (auto index : msgDigitsIndex) {
    LOG(debug) << indexcount << ":" << index;
  }
  if (msgDigitsIndex.size() != msgDigits.size()) {
    //error condition for sort.
    LOG(fatal) << "Cant index digits as index and digits differ in size, this is not permitted. Digits size=" << msgDigits.size() << " and index size=" << msgDigitsIndex.size();
  }
  //sort the digits array TODO refactor this intoa vector index sort and possibly generalise past merely digits.
  auto sortstart = std::chrono::high_resolution_clock::now();
  for (auto& trig : triggerRecords) {
    std::stable_sort(std::begin(msgDigitsIndex) + trig.getFirstEntry(), std::begin(msgDigitsIndex) + trig.getNumberOfObjects() + trig.getFirstEntry(),
                     [&msgDigits](auto&& PH1, auto&& PH2) { return digitindexcompare(PH1, PH2, msgDigits); });
  }

  mSortingTime = std::chrono::high_resolution_clock::now() - sortstart;
  LOG(debug) << "TRD Digit Sorting took " << mSortingTime.count();
  // sort from triggerRecords.getFirstEntry() to triggerRecords.getFirstEntry()+triggerRecords.getNumberOfObjects();
  //check the incoming triggerrecords:
  int triggerrecordcount = 0;
  for (auto& trig : triggerRecords) {
    LOG(debug) << "Trigger Record ; " << triggerrecordcount << " = " << trig.getFirstEntry() << " --> " << trig.getNumberOfObjects();
    triggerrecordcount++;
  }
  for (auto& trig : trackletTriggerRecords) {
    LOG(debug) << "Trigger Tracklet  Record ; " << trig.getFirstEntry() << " --> " << trig.getNumberOfObjects();
  }
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
  //TODO make them class members
  int olddetector = -1;
  int oldrow = -1;
  int oldpad = -1;
  int loopindex = 0;
  int counttrackletadditions = 0;
  int oldsize = 0;
  double trackletrate;
  unsigned long oldtrackletcount = 0;
  mTotalRawWordsWritten = 0; // words written for the raw format of 4x32bits, where 4 can be 2 to 4 depending on # of tracklets in the block.
  mOldHalfChamberLinkId = 0;
  mNewTrackletHCHeaderHasBeenWritten = false;

  // now to loop over the incoming digits.
  auto digitloopstart = std::chrono::high_resolution_clock::now();
  uint64_t digitcounter = 0;
  double b = 0;
  LOG(debug4) << "now for digit loop ";
  for (auto digititerator = msgDigitsIndex.begin(); digititerator != msgDigitsIndex.end() /* && std::distance(msgDigits.begin(),digititerator)<7*/; ++digititerator) {
    //in here we have an entire padrow which corresponds to 8 TRAPs.
    //while on a single padrow, populate data structures in the 8 trapsimulator.
    //on change of padrow
    //  fireup trapsim, do its thing with each 18 sequence of pads data that already exists inside the class from previous iterations of the loop
    LOG(debug) << "Digit iterator is : " << *digititerator;
    Digit* digit = &msgDigits[*digititerator];
    int pad = digit->getPad();
    int row = digit->getRow();
    int detector = digit->getDetector();
    int rob = mFeeParam->getROBfromPad(row, pad);
    int mcm = mFeeParam->getMCMfromPad(row, pad);
    int trdstack = Geometry::getStack(detector);
    int trdlayer = Geometry::getLayer(detector);
    int fibreside = FeeParam::instance()->getRobSide(rob);

    LOG(debug) << "calculated rob and mcm at top of loop with detector:row:pad:rob:mcm ::"
               << detector << ":" << row << ":" << pad << ":" << rob << ":" << mcm
               << " LinkId:" << LinkRecord::getHalfChamberLinkId(detector, rob) << "\t\t  SM:stack:layer:side  " << detector / 30 << ":" << trdstack << ":" << trdlayer << ":" << fibreside
               << " with ORI : " << mFeeParam->getORI(detector, rob) << " and within supermodule ori index:" << mFeeParam->getORIinSM(detector, rob);
    if (digititerator == msgDigitsIndex.begin()) { // first time in loop
      oldrow = row;
      olddetector = detector;
    }
    //Are we on a new half chamber ?
    if (mOldHalfChamberLinkId != LinkRecord::getHalfChamberLinkId(detector, rob)) {
      //     hcid= detector*2 + robpos%2;
      // new half chamber so add the header to the raw data stream.
      buildTrackletHCHeaderd(mTrackletHCHeader, detector, rob, currentTriggerRecord * 42, 4);
      //buildTrackletHCHeader(mTrackletHCHeader,sector,stack,layer,side,currentTriggerRecord*42,4);
      mOldHalfChamberLinkId = LinkRecord::getHalfChamberLinkId(detector, rob);
      // now we have a problem. We must only write the halfchamberheader if a tracklet is written i.e. if the digits for this half chamber actually produce 1 or more tracklets!
      mNewTrackletHCHeaderHasBeenWritten = false;
    }
    //figure out which trigger record from digits we are on
    if (digitcounter >= triggerRecords[currentTriggerRecord].getFirstEntry() + triggerRecords[currentTriggerRecord].getNumberOfObjects()) {
      //trigger record changed.
      //Now we know the ranges so populate the triggerrecord related to the previously block of data.
      setTriggerRecord(trackletTriggerRecords, currentTriggerRecord, trapTrackletsAccum.size());
      setTriggerRecord(rawTriggerRecords, currentTriggerRecord, mTotalRawWordsWritten);
      currentTriggerRecord++;
      LOG(debug) << "changing trigger records : " << currentTriggerRecord;
    }

    if (olddetector != detector || oldrow != row) {
      // we have gone over the pad row. //TODO ??? do we need to check for change of time as well?
      //all data is inside the 8 relavent trapsimulators
      int preivousrob = mFeeParam->getROBfromPad(oldrow, oldpad); //
      //fireup Trapsim.
      auto traploopstart = std::chrono::high_resolution_clock::now();
      unsigned long numberofusedtraps = 0;
      for (int trapcounter = 0; trapcounter < 8; trapcounter++) {
        unsigned int isinit = mTrapSimulator[trapcounter].checkInitialized();
        if (mTrapSimulator[trapcounter].isDataSet()) { //firedtraps
          //this one has been filled with data for the now previous pad row.
          auto trapsimtimerstart = std::chrono::high_resolution_clock::now();
          mTrapUsedCounter[trapcounter]++;
          numberofusedtraps++;
          mTrapSimulator[trapcounter].filter();
          mTrapSimulator[trapcounter].tracklet();

          trapTracklets = mTrapSimulator[trapcounter].getTrackletArray64(); //TODO remove the copy and send the Accumulated array into the Trapsimulator
          auto trapLabels = mTrapSimulator[trapcounter].getTrackletLabels();
          if (!mNewTrackletHCHeaderHasBeenWritten && trapTracklets.size() != 0) { // take account of the case where we have data in the trapchip adc but no tracklets
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
          //set this trap sim object to have not data (effectively) reset.
          mTrapSimulator[trapcounter].unsetData();
        } else {
          LOG(debug) << "if statement is init failed [" << trapcounter << "] PROCESSING TRAP !";
        }
        //  LOG(debug) << "Finishe MCM : : " << trapcounter;
      } //end of loop over trap chips

      //timing info
      mTrapLoopTime += std::chrono::high_resolution_clock::now() - traploopstart;
      mTrapUsedFrequency[numberofusedtraps]++;

      LOG(debug) << "Row change ... Tracklets so far: " << trapTrackletsAccum.size();
      if (mShowTrackletStats > 0) {
        if (trapTrackletsAccum.size() - oldtrackletcount > mShowTrackletStats) {
          oldtrackletcount = trapTrackletsAccum.size();
          unsigned long mcmTrackletsize = trapTrackletsAccum.size();
          mTrackletTime = std::chrono::high_resolution_clock::now() - digitloopstart;
          trackletrate = mcmTrackletsize / mTrackletTime.count();
          LOG(info) << "Getting tracklets at the rate of : " << trackletrate << " Tracklets/s ... Accumulated tracklets : " << trapTrackletsAccum.size();
        }
      }
    } //if oldetector!= detector ....
    //we are still on the same detector and row.
    //add the digits to the padrow.
    //copy pad time data into where they belong in the 8 TrapSimulators for this pad.
    int mcmoffset = -1;
    int firstrob = mFeeParam->getROBfromPad(row, 5); // 5 is arbitrary, but above the lower shared pads. so will get first rob and mcm
    int firstmcm = mFeeParam->getMCMfromPad(row, 5); // 5 for same reason
                                                     //  LOG(info) <<"calculated first rob and mcm";
    int trapindex = pad / 18;
    //check trap is initialised.
    if (!mTrapSimulator[trapindex].isDataSet()) {
      LOG(debug) << "Initialising trapsimulator for triplet (" << detector << "," << rob << ","
                 << mcm << ") as its not initialized and we need to send it some adc data.";
      mTrapSimulator[trapindex].init(mTrapConfig, detector, rob, mcm);
    }
    int adc = 20 - (pad % 18) - 1;
    std::vector<o2::MCCompLabel> tmplabels;
    auto digitslabels = digitMCLabels.getLabels(digitcounter);
    for (auto& tmplabel : digitslabels) {
      tmplabels.push_back(tmplabel);
    }
    LOG(debug) << "tmplabels for set data : " << tmplabels.size() << " and gslspan digitlabels size of : " << digitslabels.size();
    LOG(debug) << " setting data with pad=" << pad << " ti=" << trapindex + 1;
    mTrapSimulator[trapindex].setData(adc, digit->getADC(), tmplabels);

    // now take care of the case of shared pads (the whole reason for doing this pad row wise).

    if (pad % 18 == 0 || (pad + 1) % 18 == 0) { //case of pad 18 and 19 must be shared to preceding trap chip adc 1 and 0 respectively.
      adc = 20 - (pad % 18) - 1;
      if (trapindex != 0) { // avoid the case of the first trap chip
        LOG(debug) << " setting data preceding with pad=" << pad << " ti=" << trapindex - 1;
        mTrapSimulator[trapindex - 1].setData(adc, digit->getADC(), tmplabels);
      }
    }
    if ((pad - 1) % 18 == 0) { // case of pad 17 must shared to next trap chip as adc 20
                               //check trap is initialised.
      adc = 20 - (pad % 18) - 1;
      if (trapindex + 1 != 8) { // avoid the case of the last trap chip.
        LOG(debug) << " setting data proceeding with pad=" << pad << " ti=" << trapindex + 1;
        mTrapSimulator[trapindex + 1].setData(adc, digit->getADC(), tmplabels);
      }
    }

    olddetector = detector;
    oldrow = row;
    oldpad = pad;
    digitcounter++;
  } // end of loop over digits.

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
