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
// this is somewhat assuming that a DPL workflow will run on one node
#include <thread> // to detect number of hardware threads
#include <string>
#include <sstream>
#include <cmath>
#include <unistd.h> // for getppid
#include <chrono>

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
#include "DataFormatsParameters/GRPObject.h"
#include "TRDBase/Digit.h" // for the Digit type
#include "TRDSimulation/TrapSimulator.h"
#include "TRDSimulation/Digitizer.h"
#include "TRDSimulation/Detector.h" // for the Hit type
#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/FeeParam.h"
#include "TRDSimulation/TrapSimulator.h"
#include "DataFormatsTRD/TriggerRecord.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

bool DigitSortComparator(o2::trd::Digit const& a, o2::trd::Digit const& b)
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

bool DigitSortComparatorPadRow(o2::trd::Digit const& a, o2::trd::Digit const& b)
{
  //this sorts the data into pad rows, and pads with in that pad row.
  //this allows us to generate a structure of 144 pads to then pass to the trap simulator
  //taking into account the shared pads.
  int rowa = a.getRow();
  int rowb = b.getRow();
  int pada = a.getPad();
  int padb = b.getPad();
  double timea = a.getTimeStamp();
  double timeb = b.getTimeStamp();
  if (timea < timeb)
    return 1;
  else if (timea == timeb) {

    if (a.getDetector() < b.getDetector())
      return 1;
    else {
      if (a.getDetector() == b.getDetector()) {
        if (rowa < rowb)
          return 1;
        else {
          if (rowa == rowb) {
            if (pada < padb)
              return 1; //leaving in for now. so I dont have to have a 144x30 entry array for the pad data. dont have to deal with pad sorting, its going to be 1 of 144 and inserted into the required array.
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
    LOG(debug) << "mTrapConfig is valid : 0x" << hex << mTrapConfig << dec;
    return mTrapConfig;
  } else {
    LOG(debug) << "mTrapConfig is invalid : 0x" << hex << mTrapConfig << dec;

    //can I just ignore the old default and pull a new "default" from ccdb?
    //// bypass pulling in trapconfigs from ccdb, will sort out later.
    if (0) //mTrapConfigName!= std::string("default"))
    {
      // try to load the requested configuration
      loadTrapConfig();
      //calib.
    } else {
      // if we still don't have a valid TRAPconfig, we give up
      LOG(debug) << "mTrapConfig is not valid now : 0x" << hex << mTrapConfig << dec;
      LOG(warn) << "Falling back to default configuration for year<2012";
      static TrapConfig trapConfigDefault(mTrapConfigName);
      mTrapConfig = &trapConfigDefault; //new TrapConfig; //&trapConfigDefault;
      TrapConfigHandler cfgHandler(mTrapConfig);
      cfgHandler.init();
      cfgHandler.loadConfig();
    }
    LOG(debug) << "using TRAPconfig :" << mTrapConfig->getConfigName().c_str() << "." << mTrapConfig->getConfigVersion().c_str();

    // we still have to load the gain tables
    // if the gain filter is active
    return mTrapConfig;
  } // end of else from if mTrapConfig
}

void TRDDPLTrapSimulatorTask::loadTrapConfig()
{
  // try to load the specified configuration from the CCDB

  LOG(info) << "looking for TRAPconfig " << mTrapConfigName;

  // const CalTrapConfig *caltrap = dynamic_cast<const CalTrapConfig*> (GetCachedCDBObject(kIDTrapConfig));
  //TODO get trap config from OCDB/CCDB
  //pull values from CDDB incoming structure or message ??

  LOG(info) << "looking for TRAPconfig " << mTrapConfigName << __FILE__ << ":" << __LINE__;
  auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();
  LOG(info) << "looking for TRAPconfig " << mTrapConfigName << __FILE__ << ":" << __LINE__;
  ccdbmgr.setTimestamp(297595);
  LOG(info) << "looking for TRAPconfig " << mTrapConfigName << __FILE__ << ":" << __LINE__;
  mTrapConfig = ccdbmgr.get<o2::trd::TrapConfig>(mTrapConfigBaseName + std::string("/") + mTrapConfigName);
  LOG(info) << "looking for TRAPconfig " << mTrapConfigName << __FILE__ << ":" << __LINE__ << "   trapconfig is : " << &mTrapConfig;
  if (mTrapConfig == nullptr) {
    LOG(info) << "looking for TRAPconfig " << mTrapConfigName << __FILE__ << ":" << __LINE__;
    //failed to find or open or connect or something to get the trapconfig from the ccdb.
    //first check the directory listing.
    LOG(warn) << " failed to get trapconfig from ccdb with name :  " << mTrapConfigName;
    LOG(info) << "looking for TRAPconfig " << mTrapConfigName << __FILE__ << ":" << __LINE__;
  }
  LOG(info) << "looking for TRAPconfig " << mTrapConfigName << __FILE__ << ":" << __LINE__;
  //  mTrapConfig = caltrap->Get(configName); //TODO it is not clear to me how this actually comes in.
  //}
  /// else {
  //    if(mTrapConfig != nullptr){
  //      delete mTrapConfig;
  //    mTrapConfig = nullptr;
  //  }
  // mTrapConfig->LOADFROMDISKDEFAULT();
  LOG(info) << "looking for TRAPconfig " << mTrapConfigName << __FILE__ << ":" << __LINE__;
  LOG(warn) << "No TRAPconfig entry found for name of " << mTrapConfigName;
  LOG(info) << "looking for TRAPconfig " << mTrapConfigName << __FILE__ << ":" << __LINE__;
  //  }
}

void TRDDPLTrapSimulatorTask::init(o2::framework::InitContext& ic)
{
  //               LOG(debug) << "Input data file is : " << ic.options().get<std::string>("simdatasrc");
  //               LOG(info) << "simSm is : " << ic.options().get<int>("simSM");
  LOG(debug1) << "TRD Trap Simulator Device with pid of : " << ::getpid();
  mfeeparam = FeeParam::instance();
  mPrintTrackletOptions = ic.options().get<int>("printtracklets");
  mDrawTrackletOptions = ic.options().get<int>("drawtracklets");
  mShowTrackletStats = ic.options().get<int>("show-trd-trackletstats");
  mTrapConfigName = ic.options().get<std::string>("trapconfig");
  LOG(info) << "Trap Simulator Device initialising with trap config of : " << mTrapConfigName;
  //  if(mDisableTrapSimulation){
  //  //now get a trapconfig to work with.
  //    LOG(warn) << "You elected to not do a trap chip simulation and hence no trapconfig was saught";
  //  }
  //  else{
  getTrapConfig();
  LOG(info) << "Trap Simulator Device initialised ... ";
  //  }
}

void TRDDPLTrapSimulatorTask::run(o2::framework::ProcessingContext& pc)
{
  LOG(info) << "TRD Trap Simulator Device running over incoming message ...";

  // get the relevant inputs for the TrapSimulator
  // the digits are going to be sorted, we therefore need a copy of the vector rather than an object created
  // directly on the input data, the output vector however is created directly inside the message
  // memory thus avoiding copy by snapshot
  auto digitsinput = pc.inputs().get<std::vector<o2::trd::Digit>>("digitinput");
  // TODO: not clear yet whether to send the digits because the snapshot method
  // has been commented out below. Rather than using snapshot (thus a copy) on a vector
  // object, this target object should be created directly in the message memory
  //auto& digits = pc.outputs().make<std::vector<o2::trd::Digit>>(Output{"TRD", "DIGITS", 0, Lifetime::Timeframe}, digitsinput.begin(), digitsinput.end());
  std::vector<o2::trd::Digit> digits(digitsinput.begin(), digitsinput.end());
  //auto mMCLabels = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::trd::MCLabel>*>("labelinput");
  //auto mTriggerRecords = pc.inputs().get<std::vector<o2::trd::TriggerRecord>>("triggerrecords");

  LOG(debug) << "Read in Digits with size of : " << digits.size();

  //set up structures to hold the returning tracklets.
  // TODO: correct naming convention, wrong convention used for local variables
  std::vector<Tracklet> mMCMTracklets; //vector to store the retrieved tracklets from an mcm
  auto& mMCMTrackletsAccum = pc.outputs().make<std::vector<Tracklet>>(Output{"TRD", "TRACKLETS", 0, Lifetime::Timeframe});
  mMCMTracklets.reserve(30);
  mMCMTrackletsAccum.reserve(digits.size() / 3); //attempt to a. conserve mem, b. stop a vector resize

  //TODO we can ignore time, and use the triggerrecords as to defined subsets to sort. TriggerRecord has the time in it for all the digits, by creation.
  //
  std::chrono::duration<double> sortingtime{0}; ///< full timer
  auto sortstart = std::chrono::high_resolution_clock::now();

  std::sort(digits.begin(), digits.end(), DigitSortComparator);

  sortingtime = std::chrono::high_resolution_clock::now() - sortstart;

  //TODO Do we care about tracklet order so long as its time stamped.
  //TODO Dont sort, just build an index array
  //TODO sort from mTriggerRecords.getFirstEntry() to mTriggerRecords.getFirstEntry()+mTriggerRecords.getNumberOfObjects();

  int olddetector = -1;
  int oldrow = -1;
  int oldpad = -1;
  int loopindex = 0;
  int counttrackletadditions = 0;
  int oldsize = 0;
  double trackletrate;
  unsigned long oldtrackletcount = 0;
  //various timers so we can see how long things take
  std::chrono::duration<double> trapsimaccumulatedtime{0}; ///< full timer
  std::chrono::duration<double> digitlooptime{0};          ///< full timer
  std::chrono::duration<double> traplooptime{0};           ///< full timer
  std::chrono::duration<double> oldelse{0};                ///< full timer
  std::chrono::duration<double> tracklettime{0};           ///< full timer

  // now to loop over the incoming digits.
  auto digitloopstart = std::chrono::high_resolution_clock::now();

  for (auto digititerator = digits.begin(); digititerator != digits.end(); ++digititerator) {
    //in here we have an entire padrow which corresponds to 8 TRAPs.
    //while on a single padrow, populate data structures in the 8 trapsimulator.
    //on change of padrow
    //  fireup trapsim, do its thing with each 18 sequence of pads data that already exists inside the class from previous iterations of the loop
    double digittime = digititerator->getTimeStamp();
    int pad = digititerator->getPad();
    int row = digititerator->getRow();
    int detector = digititerator->getDetector();
    int rob = mfeeparam->getROBfromPad(row, pad);
    int mcm = mfeeparam->getMCMfromPad(row, pad);
    if (digititerator == digits.begin()) { // first time in loop
      oldrow = row;
      olddetector = detector;
    }
    if (olddetector != detector || oldrow != row) {
      // we haved gone over the pad row. //TODO ??? do we need to check for change of time as well?
      //all data is inside the 8 relavent trapsimulators
      rob = mfeeparam->getROBfromPad(oldrow, oldpad); //
      LOG(debug) << "processing of row,mcm"
                 << " padrow changed from " << olddetector << "," << oldrow << " to " << detector << "," << row;
      //fireup Trapsim.
      auto traploopstart = std::chrono::high_resolution_clock::now();
      for (int trapcounter = 0; trapcounter < 8; trapcounter++) {
        unsigned int isinit = mTrapSimulator[trapcounter].checkInitialized();
        LOG(debug) << "is init : " << isinit;
        if (mTrapSimulator[trapcounter].isDataSet()) { //firedtraps & (1<<trapcounter){ /}/mTrapSimulator[trapcounter].checkInitialized()){
          //this one has been filled with data for the now previous pad row.
          auto trapsimtimerstart = std::chrono::high_resolution_clock::now();

          mTrapSimulator[trapcounter].filter();
          mTrapSimulator[trapcounter].tracklet();

          mMCMTracklets = mTrapSimulator[trapcounter].getTrackletArray(); //TODO remove the copy and send the Accumulated array into the Trapsimulator

          LOG(debug) << mMCMTrackletsAccum.size() << " :: " << mMCMTracklets.size() << " count tracklet additions :  " << counttrackletadditions;
          counttrackletadditions++;
          mMCMTrackletsAccum.insert(mMCMTrackletsAccum.end(), mMCMTracklets.begin(), mMCMTracklets.end());
          trapsimaccumulatedtime += std::chrono::high_resolution_clock::now() - trapsimtimerstart;
          if (mShowTrackletStats > 0) {
            if (mMCMTrackletsAccum.size() - oldsize > mShowTrackletStats) {
              LOG(debug) << "TrapSim Accumulated tracklets: " << mMCMTrackletsAccum.size() << " :: " << mMCMTracklets.size();
              oldsize = mMCMTrackletsAccum.size();
            }
          }

          // mTrapSimulator[trapcounter].zeroSupressionMapping();

          if (mDrawTrackletOptions != 0)
            mTrapSimulator[trapcounter].draw(mDrawTrackletOptions, loopindex);
          if (mPrintTrackletOptions != 0)
            mTrapSimulator[trapcounter].print(mPrintTrackletOptions);

          loopindex++;
          //set this trap sim object to have not data (effectively) reset.
          mTrapSimulator[trapcounter].unsetData();
        } else {
          LOG(debug) << "if statement is init failed [" << trapcounter << "] PROCESSING TRAP !";
        }
      } //end of loop over trap chips
      traplooptime += std::chrono::high_resolution_clock::now() - traploopstart;
      LOG(debug) << "Row change ... Tracklets so far: " << mMCMTrackletsAccum.size();
      if (mShowTrackletStats > 0) {
        if (mMCMTrackletsAccum.size() - oldtrackletcount > mShowTrackletStats) {
          oldtrackletcount = mMCMTrackletsAccum.size();
          unsigned long mcmTrackletsize = mMCMTrackletsAccum.size();
          tracklettime = std::chrono::high_resolution_clock::now() - digitloopstart;
          trackletrate = mcmTrackletsize / tracklettime.count();
          LOG(info) << "Getting tracklets at the rate of : " << trackletrate << " Tracklets/s ... Accumulated tracklets : " << mMCMTrackletsAccum.size();
        }
      }
    } //if oldetector!= detector ....
    //we are still on the same detector and row.
    //add the digits to the padrow.
    //copy pad time data into where they belong in the 8 TrapSimulators for this pad.
    int mcmoffset = -1;
    int firstrob = mfeeparam->getROBfromPad(row, 5); // 5 is arbitrary, but above the lower shared pads. so will get first rob and mcm
    int firstmcm = mfeeparam->getMCMfromPad(row, 5); // 5 for same reason
    int trapindex = pad / 18;
    //check trap is initialised.
    if (!mTrapSimulator[trapindex].isDataSet()) {
      LOG(debug) << "Initialising trapsimulator for triplet (" << detector << "," << rob << ","
                 << mcm << ") as its not initialized and we need to send it some adc data.";
      mTrapSimulator[trapindex].init(mTrapConfig, detector, rob, mcm);
    }
    int adc = 0;
    adc = 20 - (pad % 18) - 1;
    LOG(debug) << "setting data for simulator : " << trapindex << " and adc : " << adc;
    mTrapSimulator[trapindex].setData(adc, digititerator->getADC());
    // mTrapSimulator[trapindex].printAdcDatHuman(cout);

    // now take care of the case of shared pads (the whole reason for doing this pad row wise).

    if (pad % 18 == 0 || (pad + 1) % 18 == 0) { //case of pad 18 and 19 must be shared to preceding trap chip adc 1 and 0 respectively.
                                                //check trap is initialised.

      adc = 20 - (pad % 18) - 1;
      LOG(debug) << "setting data for simulator : " << trapindex - 1 << " and adc : " << adc;
      mTrapSimulator[trapindex - 1].setData(adc, digititerator->getADC());
    }
    if ((pad - 1) % 18 == 0) { // case of pad 17 must shared to next trap chip as adc 20
                               //check trap is initialised.
      adc = 20 - (pad % 18) - 1;
      LOG(debug) << "setting data for simulator : " << trapindex + 1 << " and adc : " << adc;
      if (trapindex + 1 != 8) {
        mTrapSimulator[trapindex + 1].setData(adc, digititerator->getADC());
      }
      //else { // this is not an issue as its a shared pad, simply the sharing to the next trap chip is meaningless.
      //}
    }

    olddetector = detector;
    oldrow = row;
    oldpad = pad;
  } // end of loop over digits.

  LOG(info) << "Trap simulator found " << mMCMTrackletsAccum.size() << " tracklets from " << digits.size() << " Digits";
  if (mShowTrackletStats > 0) {
    digitlooptime = std::chrono::high_resolution_clock::now() - digitloopstart;
    LOG(info) << "Trap Simulator done \\o/ ";
    LOG(info) << "Sorting took " << sortingtime.count();
    LOG(info) << "Digit loop took : " << digitlooptime.count();
    LOG(info) << "Trapsim took : " << trapsimaccumulatedtime.count();
    LOG(info) << "Traploop took : " << traplooptime.count();
  }
  // Note: do not use snapshot for TRD/DIGITS and TRD/TRACKLETS, we can avoif the copy by allocating
  // the vectors directly in the message memory, see above

  //  pc.outputs().snapshot(Output{"TRD","TRKLABELS",0,Lifetime::Timeframe},mMCLabels);
  //pc.outputs().snapshot(Output{"TRD","TRGRRecords",0,Lifetime::Timeframe},mTriggerRecords);
}

o2::framework::DataProcessorSpec getTRDTrapSimulatorSpec()
{
  return DataProcessorSpec{"TRAP", Inputs{InputSpec{"digitinput", "TRD", "DIGITS", 0}, InputSpec{"triggerrecords", "TRD", "TRGRDIG", 0}, InputSpec{"labelinput", "TRD", "LABELS", 0}},
                           Outputs{OutputSpec{"TRD", "TRACKLETS", 0, Lifetime::Timeframe}},
                           //                                   OutputSpec{"TRD","TRGRRecords",0,Lifetime::Timeframe}},
                           //                               OutputSpec{"TRD","TRKLABELS",0, Lifetime::Timeframe},
                           AlgorithmSpec{adaptFromTask<TRDDPLTrapSimulatorTask>()},
                           Options{
                             {"show-trd-trackletstats", VariantType::Int, 25000, {"Display the accumulated size and capacity at number of track intervals"}},
                             {"trapconfig", VariantType::String, "default", {"Name of the trap config from the CCDB"}},
                             {"drawtracklets", VariantType::Int, 0, {"Bitpattern of input to TrapSimulator Draw method (be very careful) one file per track"}},
                             {"printtracklets", VariantType::Int, 0, {"Bitpattern of input to TrapSimulator print method"}}}};
};

} //end namespace trd
} //end namespace o2
