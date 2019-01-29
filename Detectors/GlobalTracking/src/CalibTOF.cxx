// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <TTree.h>
#include <cassert>

#include "FairLogger.h"
#include "TOFBase/Geo.h"

#include <TFile.h>
#include "DataFormatsParameters/GRPObject.h"
#include "ReconstructionDataFormats/PID.h"

#include "GlobalTracking/CalibTOF.h"

#include "CommonConstants/LHCConstants.h"

using namespace o2::globaltracking;

ClassImp(CalibTOF);

//______________________________________________
void CalibTOF::run(int flag)
{
  ///< running the matching

  if (!mInitDone) {
    LOG(FATAL) << "init() was not done yet";
  }

  mTimerTot.Start();

  if (flag == 0) { // LHC phase --> we will use all the entries in the tree
    while(loadTOFCalibInfo()){ // fill here all histos you need 
      fillLHCphaseCalibInput(); // we will fill the input for the LHC phase calibration
    }
    doLHCPhaseCalib();
  }
  else { // channel offset + problematic (flag = 1), or time slewing (flag = 2)
    for (int ich = 0; ich < o2::tof::Geo::NCHANNELS; ich++){
      mCurrTOFInfoTreeEntry = ich - o2::tof::Geo::NCHANNELS;
      while(loadTOFCalibInfo(o2::tof::Geo::NCHANNELS)){ // fill here all histos you need 
	fillChannelCalibInput(); // we will fill the input for the channel-level calibration
      }
      doChannelLevelCalibration(flag);
      resetChannelLevelHistos(flag);
    }
    
  }

  mTimerTot.Stop();
  printf("Timing:\n");
  printf("Total:        ");
  mTimerTot.Print();
}

//______________________________________________
void CalibTOF::init()
{
  ///< initizalizations

  if (mInitDone) {
    LOG(ERROR) << "Initialization was already done";
    return;
  }

  attachInputTrees();

  // create output branch with output -- for now this is empty
  if (mOutputTree) {
    //    mOutputTree->Branch(mOutputBranchName.data(), &mXXXXXXX);
    //    LOG(INFO) << "Matched tracks will be stored in " << mOutputBranchName << " branch of tree "
    //              << mOutputTree->GetName();
  } else {
    LOG(ERROR) << "Output tree is not attached, matched tracks will not be stored";
  }

  mInitDone = true;

  // prepare histos
  mHistoLHCphase = new TH1F("hLHCphase",";clock offset (ps)",1000,-24400,24400);

  {
    mTimerTot.Stop();
    mTimerTot.Reset();
  }

  print();
}

//______________________________________________
void CalibTOF::print() const
{
  ///< print the settings

  LOG(INFO) << "****** component for calibration of TOF channels ******";
  if (!mInitDone) {
    LOG(INFO) << "init is not done yet - nothing to print";
    return;
  }

  LOG(INFO) << "**********************************************************************";
}

//______________________________________________
void CalibTOF::attachInputTrees()
{
  ///< attaching the input tree

  if (!mTreeCollectedCalibInfoTOF) {
    LOG(FATAL) << "Input tree with collected TOF calib infos is not set";
  }

  if (!mTreeCollectedCalibInfoTOF->GetBranch(mCollectedCalibInfoTOFBranchName.data())) {
    LOG(FATAL) << "Did not find collected TOF calib info branch " << mCollectedCalibInfoTOFBranchName << " in the input tree";
  }
  mTreeCollectedCalibInfoTOF->SetBranchAddress(mCollectedCalibInfoTOFBranchName.data(), &mCalibInfoTOF);
  LOG(INFO) << "Attached tracksTOF calib info " << mCollectedCalibInfoTOFBranchName << " branch with " << mTreeCollectedCalibInfoTOF->GetEntries()
            << " entries";

  mCurrTOFInfoTreeEntry = -1;
}

//______________________________________________
bool CalibTOF::loadTOFCalibInfo(int increment)
{
  ///< load next chunk of TOF infos
  printf("Loading TOF calib infos: number of entries in tree = %lld\n", mTreeCollectedCalibInfoTOF->GetEntries());

  mCurrTOFInfoTreeEntry += increment;
  
  while (mCurrTOFInfoTreeEntry < mTreeCollectedCalibInfoTOF->GetEntries()) {
    mTreeCollectedCalibInfoTOF->GetEntry(mCurrTOFInfoTreeEntry);
    LOG(INFO) << "Loading TOF calib info entry " << mCurrTOFInfoTreeEntry << " -> " << mCalibInfoTOF->size()<< " infos";
    
    if (!mCalibInfoTOF->size()) {
      mCurrTOFInfoTreeEntry += increment;
      continue;
    }
    return true;
  }
  mCurrTOFInfoTreeEntry -= increment;

  return false;
}
//______________________________________________
int CalibTOF::doCalib(int flag, int channel)
{

  

  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1./bc;

  int status = 0;

  // implemented for flag=0, channel=-1 (-1 means all!)
  for(auto infotof = mCalibInfoTOF->begin(); infotof != mCalibInfoTOF->end(); infotof++){
    double dtime = infotof->getDeltaTimePi();
    dtime -= int(dtime*bc_inv + 0.5)*bc;

    mHistoLHCphase->Fill(dtime);
  }
  return status;
}

