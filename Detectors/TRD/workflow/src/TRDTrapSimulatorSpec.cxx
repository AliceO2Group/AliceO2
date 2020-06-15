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
#include "TChain.h"

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
#include <SimulationDataFormat/MCTruthContainer.h>
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

//#ifdef WITH_OPENMP
//#include <omp.h>
//#endif ci is failing on this, sort out another time.

using namespace o2::framework;

namespace o2
{
namespace trd
{

bool msgDigitSortComparator(o2::trd::Digit const& a, o2::trd::Digit const& b)
{
  FeeParam* fee = FeeParam::instance();
  int rowa = a.getRow();
  int rowb = b.getRow();
  int pada = a.getPad();
  int padb = b.getPad();
  double timea = a.getTimeStamp();
  double timeb = b.getTimeStamp();
  int roba = fee->getROBfromPad(rowa, pada);
  int robb = fee->getROBfromPad(rowb, padb);
  int mcma = fee->getMCMfromPad(rowa, pada);
  int mcmb = fee->getMCMfromPad(rowb, padb);
  //LOG(info) << "comparing " << rowa << ":" << pada <<":" << roba <<" "<< mcma  << " with " << rowb << ":" << padb <<":" << robb <<" "<< mcmb;
  if (timea < timeb) {
    //  LOG(info) << "yip timea < timeb " << timea <<"<" << timeb;
    return 1;
  } else if (timea == timeb) {

    if (a.getDetector() < b.getDetector())
      return 1;
    else {
      if (a.getDetector() == b.getDetector()) {
        if (roba < robb)
          return 1;
        else {
          if (roba == robb) {
            if (mcma < mcmb)
              return 1;
            else
              return 0;
          } else
            return 0;
        }
        return 0;
      }
      return 0;
    }
    return 0;
  }
  return 0;
}

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
    LOG(debug) << "using TRAPconfig :" << mTrapConfig->getConfigName().c_str() << "." << mTrapConfig->getConfigVersion().c_str();

    // we still have to load the gain tables
    // if the gain filter is active
    return mTrapConfig;
  } // end of else from if mTrapConfig
}

void PrintDmemValue3(TrapConfig::TrapDmemWord* trapval, std::ofstream& output)
{
  output << "\t AllocationMode : " << trapval->getAllocMode() << std::endl;
  output << "\t Array size : " << trapval->getDataSize() << std::endl;
  for (int dataarray = 0; dataarray < trapval->getDataSize(); dataarray++) {
    output << "\t " << trapval->getDataRaw(dataarray) << " : valid : " << trapval->getValidRaw(dataarray) << std::endl;
  }
}
void PrintRegisterValue3(TrapConfig::TrapRegister* trapval, std::ofstream& output)
{
  output << "\t AllocationMode : " << trapval->getAllocMode() << std::endl;
  output << "\t Array size : " << trapval->getDataSize() << std::endl;
  for (int dataarray = 0; dataarray < trapval->getDataSize(); dataarray++) {
    output << "\t " << trapval->getDataRaw(dataarray) << " : valid : " << trapval->getValidRaw(dataarray) << std::endl;
  }
}
void PrintTrapConfigAsStored3(TrapConfig* trapconfig)
{
  std::ofstream run3config("run3trapconfig-AsStored-insidedpl.txt");
  run3config << "Trap Registers : " << std::endl;
  for (int regvalue = 0; regvalue < TrapConfig::kLastReg; regvalue++) {
    run3config << " Trap : " << trapconfig->mRegisterValue[regvalue].getName()
               << " at : 0x " << std::hex << trapconfig->mRegisterValue[regvalue].getAddr() << std::dec
               << " with nbits : " << trapconfig->mRegisterValue[regvalue].getNbits()
               << " and reset value of : " << trapconfig->mRegisterValue[regvalue].getResetValue() << std::endl;
    // now for the inherited AliTRDtrapValue members;
    PrintRegisterValue3(&trapconfig->mRegisterValue[regvalue], run3config);
  }

  //  run3config << "done with regiser values now for dmemwords" << std::endl;
  run3config << "DMEM Words : " << std::endl;
  for (int dmemwords = 0; dmemwords < TrapConfig::mgkDmemWords; dmemwords++) {
    // copy fName, fAddr
    // inherited from trapvalue : fAllocMode, fSize fData and fValid
    //        trapconfig->mDmem[dmemwords].mName= run2config->fDmem[dmemwords].fName; // this gets set on setting the address
    run3config << "Name : " << trapconfig->mDmem[dmemwords].getName() << " :address : " << trapconfig->mDmem[dmemwords].getAddress() << std::endl;
    PrintDmemValue3(&trapconfig->mDmem[dmemwords], run3config);
  }
}

void TRDDPLTrapSimulatorTask::loadTrapConfig()
{
  // try to load the specified configuration from the CCDB

  LOG(debug) << "looking for TRAPconfig " << mTrapConfigName;

  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbmgr.setTimestamp(297595);
  //default is : mTrapConfigName="cf_pg-fpnp32_zs-s16-deh_tb30_trkl-b5n-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5549";
  mTrapConfigName = "c";
  mTrapConfig = ccdbmgr.get<o2::trd::TrapConfig>("TRD_test/TrapConfig2020/c");
  if (mTrapConfig == nullptr) {
    //failed to find or open or connect or something to get the trapconfig from the ccdb.
    //first check the directory listing.
    LOG(warn) << " failed to get trapconfig from ccdb with name :  " << mTrapConfigName;
    LOG(info) << "Retrieving trapconfig failed";
  } else {
    //TODO figure out how to get the debug level from logger and only do this for debug option to --severity debug (or what ever the command actualy is)
    if (mEnableTrapConfigDump)
      PrintTrapConfigAsStored3(mTrapConfig);
  }
}

void TRDDPLTrapSimulatorTask::setOnlineGainTables()
{
  const int nDets = 540; //TRDGeometry::Ndet();
  const int nMcms = TRDGeometry::MCMmax();
  const int nChs = TRDGeometry::ADCmax();
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
      const int nRobs = TRDGeometry::getStack(iDet) == 2 ? TRDGeometry::ROBmaxC0() : TRDGeometry::ROBmaxC1();
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
  mFeeParam = FeeParam::instance();
  mPrintTrackletOptions = ic.options().get<int>("trd-printtracklets");
  mDrawTrackletOptions = ic.options().get<int>("trd-drawtracklets");
  mShowTrackletStats = ic.options().get<int>("show-trd-trackletstats");
  mTrapConfigName = ic.options().get<std::string>("trd-trapconfig");
  mPrintOutTrapConfig = ic.options().get<bool>("trd-printtrapconfig");
  mDebugRejectedTracklets = ic.options().get<bool>("trd-debugrejectedtracklets");
  mEnableOnlineGainCorrection = ic.options().get<bool>("trd-onlinegaincorrection");
  mOnlineGainTableName = ic.options().get<std::string>("trd-onlinegaintable");
  mRunNumber = ic.options().get<int>("trd-runnum");
  mEnableTrapConfigDump = ic.options().get<bool>("trd-dumptrapconfig");
  mFixTriggerRecords = ic.options().get<bool>("trd-fixtriggerrecord");
  //Connect to CCDB for all things needing access to ccdb.
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

  const o2::trd::Digit *a, *b;
  a = &originalDigits[A];
  b = &originalDigits[B];
  FeeParam* fee = FeeParam::instance();
  int rowa = a->getRow();
  int rowb = b->getRow();
  int pada = a->getPad();
  int padb = b->getPad();
  double timea = a->getTimeStamp();
  double timeb = b->getTimeStamp();
  int roba = fee->getROBfromPad(rowa, pada);
  int robb = fee->getROBfromPad(rowb, padb);
  int mcma = fee->getMCMfromPad(rowa, pada);
  int mcmb = fee->getMCMfromPad(rowb, padb);
  if (a->getTimeStamp() < b->getTimeStamp())
    return 1;
  if (a->getTimeStamp() > b->getTimeStamp())
    return 0;
  // timestamps are equal
  if (a->getDetector() < b->getDetector())
    return 1;
  if (a->getDetector() > b->getDetector())
    return 0;
  //detectors are equal
  if (roba < robb)
    return 1;
  if (roba > robb)
    return 0;
  //rob is equal
  if (mcma < mcmb)
    return 1;
  else
    return 0;
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
  //TODO need to change to gsl::span.

  // get inputs for the TrapSimulator
  // the digits are going to be sorted, we therefore need a copy of the vector rather than an object created
  // directly on the input data, the output vector however is created directly inside the message
  // memory thus avoiding copy by snapshot
  auto msgDigits = pc.inputs().get<std::vector<o2::trd::Digit>>("digitinput");
  //  auto digits pc.outputs().make<std::vector<o2::trd::Digit>>(Output{"TRD", "TRKDIGITS", 0, Lifetime::Timeframe}, msgDigits.begin(), msgDigits.end());
  auto digitMCLabels = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labelinput");

  //  auto rawDataOut = pc.outputs().make<char>(Output{"TRD", "RAWDATA", 0, Lifetime::Timeframe}, 1000); //TODO number is just a place holder until we start using it.
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> trackletMCLabels;

  auto triggerRecords = pc.inputs().get<std::vector<o2::trd::TriggerRecord>>("triggerrecords");
  uint64_t currentTriggerRecord = 0;

  for (auto& trig : triggerRecords) {
    LOG(debug) << "Trigger Record ; " << trig.getFirstEntry() << " --> " << trig.getNumberOfObjects();
  }
  // fix incoming trigger records if requested.
  if (mFixTriggerRecords)
    fixTriggerRecords(triggerRecords);

  std::vector<o2::trd::TriggerRecord> trackletTriggerRecords = triggerRecords; // copy over the whole thing but we only really want the bunch crossing info.

  //TODO these must be created directly in the output as done at the top of this run method
  std::vector<unsigned int> msgDigitsIndex;
  msgDigitsIndex.reserve(msgDigits.size());

  LOG(debug) << "Read in msgDigits with size of : " << msgDigits.size() << " labels contain : " << digitMCLabels->getNElements() << " with and index size of  : " << digitMCLabels->getIndexedSize();

  if (digitMCLabels->getIndexedSize() != msgDigits.size()) {
    LOG(debug) << "Read in msgDigits with size of : " << msgDigits.size() << " labels contain : " << digitMCLabels->getNElements() << " with and index size of  : " << digitMCLabels->getIndexedSize();
    LOG(warn) << "Digits and Labels coming into TrapSimulator are of differing sizes, labels will be jibberish. ";
  }
  //set up structures to hold the returning tracklets.
  std::vector<Tracklet> trapTracklets; //vector to store the retrieved tracklets from an trapsim object
  std::vector<Tracklet> trapTrackletsAccum;
  trapTracklets.reserve(30);
  trapTrackletsAccum.reserve(msgDigits.size() / 3);
  msgDigitsIndex.reserve(msgDigits.size());

  int count = 0;
  //make msgDigitsIndex a simple vector of ascending numbers mapping trivially into the msgDigits vector.
  for (int i = 0; i < msgDigits.size(); i++) {
    msgDigitsIndex.push_back(i);
  }
  LOG(debug) << "msgdigitsindex is " << msgDigitsIndex.size();

  auto sortstart = std::chrono::high_resolution_clock::now();
  //sort the digits array
  std::stable_sort(msgDigits.begin(), msgDigits.end(), msgDigitSortComparator);
  mSortingTime = std::chrono::high_resolution_clock::now() - sortstart;
  LOG(info) << "TRD Digit Sorting took " << mSortingTime.count();
  // sort from triggerRecords.getFirstEntry() to triggerRecords.getFirstEntry()+triggerRecords.getNumberOfObjects();
  //check the incoming triggerrecords:
  for (auto& trig : triggerRecords) {
    LOG(debug) << "Trigger Record ; " << trig.getFirstEntry() << " --> " << trig.getNumberOfObjects();
  }
  for (auto& trig : trackletTriggerRecords) {
    LOG(debug) << "Trigger Tracklet  Record ; " << trig.getFirstEntry() << " --> " << trig.getNumberOfObjects();
  }
  //accounting variables for various things.
  int olddetector = -1;
  int oldrow = -1;
  int oldpad = -1;
  int loopindex = 0;
  int counttrackletadditions = 0;
  int oldsize = 0;
  double trackletrate;
  unsigned long oldtrackletcount = 0;

  // now to loop over the incoming digits.
  auto digitloopstart = std::chrono::high_resolution_clock::now();
  uint64_t digitcounter = 0;
  for (auto digititerator = msgDigits.begin(); digititerator != msgDigits.end() /* && loopindex<300*/; ++digititerator) {
    //in here we have an entire padrow which corresponds to 8 TRAPs.
    //while on a single padrow, populate data structures in the 8 trapsimulator.
    //on change of padrow
    //  fireup trapsim, do its thing with each 18 sequence of pads data that already exists inside the class from previous iterations of the loop
    double digittime = digititerator->getTimeStamp();
    int pad = digititerator->getPad();
    int row = digititerator->getRow();
    int detector = digititerator->getDetector();
    int rob = mFeeParam->getROBfromPad(row, pad);
    int mcm = mFeeParam->getMCMfromPad(row, pad);
    LOG(debug3) << "calculated rob and mcm at top of loop with detector:row:pad:rob:mcm" << detector << ":" << row << ":" << pad << ":" << rob << ":" << mcm;
    if (digititerator == msgDigits.begin()) { // first time in loop
      oldrow = row;
      olddetector = detector;
    }

    //figure out which trigger record from digits we are on
    if (digitcounter >= triggerRecords[currentTriggerRecord].getFirstEntry() + triggerRecords[currentTriggerRecord].getNumberOfObjects()) {
      //trigger record changed.
      // so increment the tracklet trigger records and fill accordingly for the now completed prior triggerrecord.
      uint64_t triggerrecordstart = 0;
      if (currentTriggerRecord == 0) { // for not the first one we can simply look back to the previous one to get the start.
        triggerrecordstart = 0;
        trackletTriggerRecords[currentTriggerRecord].setDataRange(triggerrecordstart, trapTrackletsAccum.size());
      } else {
        triggerrecordstart = trackletTriggerRecords[currentTriggerRecord - 1].getFirstEntry() + trackletTriggerRecords[currentTriggerRecord - 1].getNumberOfObjects();
        trackletTriggerRecords[currentTriggerRecord].setDataRange(triggerrecordstart, trapTrackletsAccum.size() - triggerrecordstart);
      }
      currentTriggerRecord++; //move to next trigger record.
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
        // LOG(debug3) << "Start of trap  : " << trapcounter;
        if (mTrapSimulator[trapcounter].isDataSet()) { //firedtraps
          //this one has been filled with data for the now previous pad row.
          auto trapsimtimerstart = std::chrono::high_resolution_clock::now();
          mTrapUsedCounter[trapcounter]++;
          numberofusedtraps++;
          mTrapSimulator[trapcounter].filter();
          mTrapSimulator[trapcounter].tracklet();

          trapTracklets = mTrapSimulator[trapcounter].getTrackletArray(); //TODO remove the copy and send the Accumulated array into the Trapsimulator
          auto trapLabels = mTrapSimulator[trapcounter].getTrackletLabels();

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

          // if (mDrawTrackletOptions != 0)
          //  mTrapSimulator[trapcounter].draw(mDrawTrackletOptions, loopindex);
          if (mDebugRejectedTracklets) {                    //&& trapTracklets.size()==0) {
            mTrapSimulator[trapcounter].draw(7, loopindex); //draw adc when no tracklets are found.A
            LOG(info) << "loop index  : " << loopindex;
            mTrapSimulator[trapcounter].print(1);
            // if(loopindex==320) LOG(fatal) <<"exiting at trap loop count 320";
          }
          //          if (mPrintTrackletOptions != 0)
          //            mTrapSimulator[trapcounter].print(mPrintTrackletOptions);

          loopindex++;
          //set this trap sim object to have not data (effectively) reset.
          mTrapSimulator[trapcounter].unsetData();
        } else {
          LOG(debug) << "if statement is init failed [" << trapcounter << "] PROCESSING TRAP !";
        }
        //  LOG(info) << "Finishe MCM : : " << trapcounter;
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
      //  LOG(debug) << "Initialising trapsimulator for triplet (" << detector << "," << rob << ","
      //             << mcm << ") as its not initialized and we need to send it some adc data.";
      mTrapSimulator[trapindex].init(mTrapConfig, detector, rob, mcm);
    }
    int adc = 0;
    adc = 20 - (pad % 18) - 1;
    std::vector<o2::MCCompLabel> tmplabels;
    auto digitslabels = digitMCLabels->getLabels(digitcounter);
    for (auto& tmplabel : digitslabels) {
      tmplabels.push_back(tmplabel);
    }
    LOG(debug) << "tmplabels for set data : " << tmplabels.size() << " and gslspan digitlabels size of : " << digitslabels.size();
    mTrapSimulator[trapindex].setData(adc, digititerator->getADC(), tmplabels);

    // now take care of the case of shared pads (the whole reason for doing this pad row wise).

    if (pad % 18 == 0 || (pad + 1) % 18 == 0) { //case of pad 18 and 19 must be shared to preceding trap chip adc 1 and 0 respectively.
      adc = 20 - (pad % 18) - 1;
      mTrapSimulator[trapindex - 1].setData(adc, digititerator->getADC(), tmplabels);
    }
    if ((pad - 1) % 18 == 0) { // case of pad 17 must shared to next trap chip as adc 20
                               //check trap is initialised.
      adc = 20 - (pad % 18) - 1;
      if (trapindex + 1 != 8) { // avoid the case of the last trap chip.
        mTrapSimulator[trapindex + 1].setData(adc, digititerator->getADC(), tmplabels);
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
  LOG(info) << "Trap simulator found " << trapTrackletsAccum.size() << " tracklets from " << msgDigits.size() << " Digits and " << trackletMCLabels.getIndexedSize() << " associated MC Label indexes and " << trackletMCLabels.getNElements() << " associated MC Labels";
  if (mShowTrackletStats > 0) {
    mDigitLoopTime = std::chrono::high_resolution_clock::now() - digitloopstart;
    LOG(info) << "Trap Simulator done \\o/ ";
#ifdef TRDTIMESORT
    LOG(info) << "Sorting took " << mSortingTime.count();
#endif
    LOG(info) << "Digit loop took : " << mDigitLoopTime.count();
    LOG(info) << "Trapsim took : " << mTrapSimAccumulatedTime.count();
    LOG(info) << "Traploop took : " << mTrapLoopTime.count();
    for (auto trapcount : mTrapUsedFrequency) {
      LOG(info) << "# traps fired Traploop are : " << trapcount;
    }
    for (auto trapcount : mTrapUsedCounter) {
      LOG(info) << "each trap position fired   : " << trapcount;
    }
  }

  pc.outputs().snapshot(Output{"TRD", "TRACKLETS", 0, Lifetime::Timeframe}, trapTrackletsAccum);
  pc.outputs().snapshot(Output{"TRD", "TRKTRGRD", 0, Lifetime::Timeframe}, trackletTriggerRecords);
  /*pc.outputs().snapshot(Output{"TRD", "TRKLABELS", 0, Lifetime::Timeframe}, trackletMCLabels);  */
  // LOG(info) << "digit MCLabels is of type : " << type_id_with_cvr<decltype(digitMCLabels)>().pretty_name();
  LOG(info) << "exiting the trap sim run method ";
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec()
{
  return DataProcessorSpec{"TRAP", Inputs{InputSpec{"digitinput", "TRD", "DIGITS", 0}, InputSpec{"triggerrecords", "TRD", "TRGRDIG", 0}, InputSpec{"labelinput", "TRD", "LABELS", 0}},

                           Outputs{OutputSpec{"TRD", "TRACKLETS", 0, Lifetime::Timeframe},
                                   OutputSpec{"TRD", "TRKTRGRD", 0, Lifetime::Timeframe}
                                   /*OutputSpec{"TRD", "TRKDIGITS", 0, Lifetime::Timeframe},*/
                                   /*OutputSpec{"TRD", "TRKLABELS", 0, Lifetime::Timeframe},*/
                                   /*OutputSpec{"TRD", "RAWDATA", 0, Lifetime::Timeframe}*/},
                           AlgorithmSpec{adaptFromTask<TRDDPLTrapSimulatorTask>()},
                           Options{
                             {"show-trd-trackletstats", VariantType::Int, 25000, {"Display the accumulated size and capacity at number of track intervals"}},
                             {"trd-trapconfig", VariantType::String, "default", {"Name of the trap config from the CCDB"}},
                             {"trd-printtrapconfig", VariantType::Bool, false, {"Name of the trap config from the CCDB"}},
                             {"trd-drawtracklets", VariantType::Int, 0, {"Bitpattern of input to TrapSimulator Draw method (be very careful) one file per track"}},
                             {"trd-printtracklets", VariantType::Int, 0, {"Bitpattern of input to TrapSimulator print method"}},
                             {"trd-fixtriggerrecord", VariantType::Bool, false, {"Fix trigger record alignment, temporary, hence false by default"}},
                             {"trd-onlinegaincorrection", VariantType::Bool, false, {"Apply online gain calibrations, mostly for back checking to run2 by setting FGBY to 0"}},
                             {"trd-onlinegaintable", VariantType::String, "Krypton_2015-02", {"Online gain table to be use, names found in CCDB, obviously trd-onlinegaincorrection must be set as well."}},
                             {"trd-debugrejectedtracklets", VariantType::Bool, false, {"Output all MCM where tracklets were not identified"}},
                             {"trd-dumptrapconfig", VariantType::Bool, false, {"Dump the trapconfig at loading"}},
                             {"trd-runnum", VariantType::Int, 297595, {"Run number to use to anchor simulation to, defaults to 297595"}}}};
};

} //end namespace trd
} //end namespace o2
