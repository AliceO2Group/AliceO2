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
    while(loadTOFCollectedCalibInfo()){ // fill here all histos you need 
      fillLHCphaseCalibInput(); // we will fill the input for the LHC phase calibration
    }
    doLHCPhaseCalib();
    mLHCphase=mFuncLHCphase->GetParameter(1);
    mLHCphaseErr=mFuncLHCphase->GetParError(1);
  }
  //  else { // channel offset + problematic (flag = 1), or time slewing (flag = 2)
  if(flag == 0){ // for the moment compute everything idependetly of the flag
    for (int ich = 0; ich < o2::tof::Geo::NCHANNELS/10; ich++){
      if(!(ich%100)) printf("channel %i\n",ich);
      mCurrTOFInfoTreeEntry = ich - o2::tof::Geo::NCHANNELS;
      while(loadTOFCollectedCalibInfo(o2::tof::Geo::NCHANNELS)){ // fill here all histos you need 
	fillChannelCalibInput(mInitialCalibChannelOffset[ich]); // we will fill the input for the channel-level calibration
      }
      if(mHistoChOffsetTemp->GetEntries() > 30){
	doChannelLevelCalibration(flag);
	mCalibChannelOffset[ich] = mFuncChOffset->GetParameter(1) + mInitialCalibChannelOffset[ich];
	mCalibChannelOffsetErr[ich] = mFuncChOffset->GetParError(1);
      }
      resetChannelLevelHistos(flag);
    }
    
  }

  mOutputTree->Fill();

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

    std::fill_n(mCalibChannelOffset, o2::tof::Geo::NCHANNELS, 0);
    std::fill_n(mCalibChannelOffsetErr, o2::tof::Geo::NCHANNELS, 0);
    std::fill_n(mInitialCalibChannelOffset, o2::tof::Geo::NCHANNELS, 0);

    // load better knoldge of channel offset (from CCDB?)
    // to be done

  // create output branch with output -- for now this is empty
  if (mOutputTree) {
    mOutputTree->Branch("LHCphase", &mLHCphase,"LHCphase/F");
    mOutputTree->Branch("LHCphaseErr", &mLHCphaseErr,"LHCphaseErr/F");
    mOutputTree->Branch("nChannels", &mNChannels,"nChannels/I");
    mOutputTree->Branch("ChannelOffset", mCalibChannelOffset,"ChannelOffset[nChannels]/F");
    mOutputTree->Branch("ChannelOffsetErr", mCalibChannelOffsetErr,"ChannelOffsetErr[nChannels]/F");
    //    LOG(INFO) << "Matched tracks will be stored in " << mOutputBranchName << " branch of tree "
    //              << mOutputTree->GetName();
  } else {
    LOG(ERROR) << "Output tree is not attached, matched tracks will not be stored";
  }

  mInitDone = true;

  // prepare histos
  mHistoLHCphase = new TH1F("hLHCphase",";clock offset (ps)",1000,-24400,24400);
  mHistoChOffsetTemp  = new TH1F("hLHCchOffsetTemp",";channel offset (ps)",1000,-24400,24400);

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
  /*
  LOG(INFO) << "Attached tracksTOF calib info " << mCollectedCalibInfoTOFBranchName << " branch with " << mTreeCollectedCalibInfoTOF->GetEntries()
            << " entries";
  */
  mCurrTOFInfoTreeEntry = -1;
}

//______________________________________________
bool CalibTOF::loadTOFCollectedCalibInfo(int increment)
{
  ///< load next chunk of TOF infos
  //  printf("Loading TOF calib infos: number of entries in tree = %lld\n", mTreeCollectedCalibInfoTOF->GetEntries());

  mCurrTOFInfoTreeEntry += increment;
  while (mCurrTOFInfoTreeEntry < mTreeCollectedCalibInfoTOF->GetEntries()
    && mCurrTOFInfoTreeEntry < o2::tof::Geo::NCHANNELS) {
    mTreeCollectedCalibInfoTOF->GetEntry(mCurrTOFInfoTreeEntry);
    //    LOG(INFO) << "Loading TOF calib info entry " << mCurrTOFInfoTreeEntry << " -> " << mCalibInfoTOF->size()<< " infos";
    
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
}
//______________________________________________

void CalibTOF::fillLHCphaseCalibInput(){
  // we will fill the input for the LHC phase calibration
  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1./bc;
    
  // implemented for flag=0, channel=-1 (-1 means all!)
  for(auto infotof = mCalibInfoTOF->begin(); infotof != mCalibInfoTOF->end(); infotof++){
    double dtime = infotof->getDeltaTimePi();
    dtime -= int(dtime*bc_inv + 0.5)*bc;
    
    mHistoLHCphase->Fill(dtime);
  }
  
}
//______________________________________________

void CalibTOF::doLHCPhaseCalib(){
  // calibrate with respect LHC phase
  if(!mFuncLHCphase){
    mFuncLHCphase = new TF1("fLHCphase","gaus");
    mFuncLHCphase->SetParameter(0,1000);
    mFuncLHCphase->SetParameter(1,0);
    mFuncLHCphase->SetParameter(2,200);
    mFuncLHCphase->SetParLimits(2,100,400);
  }
  mHistoLHCphase->Fit(mFuncLHCphase,"WW","Q");
  mHistoLHCphase->Fit(mFuncLHCphase,"","Q",mFuncLHCphase->GetParameter(1)-600,mFuncLHCphase->GetParameter(1)+400);
}
//______________________________________________

void CalibTOF::fillChannelCalibInput(float offset){
  // we will fill the input for the channel-level calibration
  // we will fill the input for the LHC phase calibration
  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1./bc;
    
  // implemented for flag=0, channel=-1 (-1 means all!)
  for(auto infotof = mCalibInfoTOF->begin(); infotof != mCalibInfoTOF->end(); infotof++){
    double dtime = infotof->getDeltaTimePi();
    dtime -= int(dtime*bc_inv + 0.5)*bc;
    
    mHistoChOffsetTemp->Fill(dtime);
  }
}
//______________________________________________

void CalibTOF::doChannelLevelCalibration(int flag){
  // calibrate single channel from histos
   if(!mFuncChOffset){
    mFuncChOffset = new TF1("fLHCchOffset","gaus");
    mFuncChOffset->SetParLimits(2,100,400);
  }
  mFuncChOffset->SetParameter(0,100);
  mFuncChOffset->SetParameter(1,mHistoChOffsetTemp->GetMean());
  mFuncChOffset->SetParameter(2,200);

  mHistoChOffsetTemp->Fit(mFuncChOffset,"WW","Q"); 
  mHistoChOffsetTemp->Fit(mFuncChOffset,"","Q",mFuncChOffset->GetParameter(1)-600,mFuncChOffset->GetParameter(1)+400); 
}
//______________________________________________

void CalibTOF::resetChannelLevelHistos(int flag){
  // reset signle channel histos
  mHistoChOffsetTemp->Reset();
}

