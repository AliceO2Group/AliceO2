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
#include "Field/MagneticField.h"
#include "Field/MagFieldFast.h"
#include "TOFBase/Geo.h"

#include <TFile.h>
#include "DataFormatsParameters/GRPObject.h"
#include "ReconstructionDataFormats/PID.h"

#include "GlobalTracking/CalibTOF.h"

#include "CommonConstants/LHCConstants.h"

using namespace o2::globaltracking;

ClassImp(CalibTOF);

//______________________________________________
void CalibTOF::run()
{
  ///< running the matching

  if (!mInitDone) {
    LOG(FATAL) << "init() was not done yet";
  }

  mTimerTot.Start();
  
  while(loadTOFCalibInfo()){ // fill here all histos you need 
    // ...
    doCalib(0, -1); // flag: 0=LHC phase, 1=channel offset+problematic(return value), 2=time-slewing

  }

  // fit and extract calibration parameters once histos are filled
  // ...

#ifdef _ALLOW_DEBUG_TREES_
  if(mDBGFlags)
    mDBGOut.reset();
#endif

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

  // create output branch with track-tof matching
  if (mOutputTree) {
    //    mOutputTree->Branch(mOutputBranchName.data(), &mXXXXXXX);
    //    LOG(INFO) << "Matched tracks will be stored in " << mOutputBranchName << " branch of tree "
    //              << mOutputTree->GetName();
  } else {
    LOG(ERROR) << "Output tree is not attached, matched tracks will not be stored";
  }

#ifdef _ALLOW_DEBUG_TREES_
  // debug streamer
  if (mDBGFlags) {
    mDBGOut = std::make_unique<o2::utils::TreeStreamRedirector>(mDebugTreeFileName.data(), "recreate");
  }
#endif

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

  if (!mTreeTOFCalibInfo) {
    LOG(FATAL) << "Input tree with TOF calib infos is not set";
  }

  if (!mTreeTOFCalibInfo->GetBranch(mTOFCalibInfoBranchName.data())) {
    LOG(FATAL) << "Did not find TOF calib info branch " << mTOFCalibInfoBranchName << " in the input tree";
  }
  mTreeTOFCalibInfo->SetBranchAddress(mTOFCalibInfoBranchName.data(), &mTOFCalibInfo);
  LOG(INFO) << "Attached tracksTOF calib info " << mTOFCalibInfoBranchName << " branch with " << mTreeTOFCalibInfo->GetEntries()
            << " entries";

  mCurrTOFInfoTreeEntry = -1;
}

//______________________________________________
bool CalibTOF::loadTOFCalibInfo()
{
  ///< load next chunk of TOF infos
  printf("Loading TOF calib infos: number of entries in tree = %lld\n", mTreeTOFCalibInfo->GetEntries());

  while (++mCurrTOFInfoTreeEntry < mTreeTOFCalibInfo->GetEntries()) {
    mTreeTOFCalibInfo->GetEntry(mCurrTOFInfoTreeEntry);
    LOG(INFO) << "Loading TOF calib info entry " << mCurrTOFInfoTreeEntry << " -> " << mTOFCalibInfo->size()<< " infos";

    if (!mTOFCalibInfo->size()) {
      continue;
    }
    return true;
  }
  --mCurrTOFInfoTreeEntry;

  return false;
}
//______________________________________________
int CalibTOF::doCalib(int flag, int channel)
{
  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1./bc;

  int status = 0;

  // implemented for flag=0, channel=-1 (-1 means all!)
  for(auto infotof=mTOFCalibInfo->begin(); infotof != mTOFCalibInfo->end(); infotof++){
    double dtime = infotof->getDeltaTimePi();
    dtime -= int(dtime*bc_inv + 0.5)*bc;

    mHistoLHCphase->Fill(dtime);
  }
  return status;
}

#ifdef _ALLOW_DEBUG_TREES_
//______________________________________________
void CalibTOF::setDebugFlag(UInt_t flag, bool on)
{
  ///< set debug stream flag
  if (on) {
    mDBGFlags |= flag;
  } else {
    mDBGFlags &= ~flag;
  }
}
/*
//_________________________________________________________
void CalibTOF::fillTOFmatchTree(const char* trname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, o2::dataformats::TrackTPCITS& trk, float intLength, float intTimePion, float timeTOF)
{
  ///< fill debug tree for TOF tracks matching check

  mTimerDBG.Start(false);

  //  Printf("************** Filling the debug tree with %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %f, %f, %f", cacheTOF, sectTOF, plateTOF, stripTOF, padXTOF, padZTOF, cacheeTrk, crossedStrip, sectPropagation, platePropagation, stripPropagation, padXPropagation, padZPropagation, resX, resZ, res);

  if(mDBGFlags){
    (*mDBGOut) << trname
	       << "clusterTOF=" << cacheTOF << "sectTOF=" << sectTOF << "plateTOF=" << plateTOF << "stripTOF=" << stripTOF << "padXTOF=" << padXTOF << "padZTOF=" << padZTOF
	       << "crossedStrip=" << crossedStrip << "sectPropagation=" << sectPropagation << "platePropagation=" << platePropagation << "stripPropagation=" << stripPropagation << "padXPropagation=" << padXPropagation
	       << "resX=" << resX << "resZ=" << resZ << "res=" << res << "track=" << trk << "intLength=" << intLength << "intTimePion=" << intTimePion << "timeTOF=" << timeTOF << "\n";
  }
  mTimerDBG.Stop();
}

//_________________________________________________________
void CalibTOF::fillTOFmatchTreeWithLabels(const char* trname, int cacheTOF, int sectTOF, int plateTOF, int stripTOF, int padXTOF, int padZTOF, int cacheeTrk, int crossedStrip, int sectPropagation, int platePropagation, int stripPropagation, int padXPropagation, int padZPropagation, float resX, float resZ, float res, o2::dataformats::TrackTPCITS& trk, int TPClabelTrackID, int TPClabelEventID, int TPClabelSourceID, int ITSlabelTrackID, int ITSlabelEventID, int ITSlabelSourceID, int TOFlabelTrackID0, int TOFlabelEventID0, int TOFlabelSourceID0, int TOFlabelTrackID1, int TOFlabelEventID1, int TOFlabelSourceID1, int TOFlabelTrackID2, int TOFlabelEventID2, int TOFlabelSourceID2, float intLength, float intTimePion, float timeTOF)
{
  ///< fill debug tree for TOF tracks matching check

  mTimerDBG.Start(false);

  if(mDBGFlags){
    (*mDBGOut) << trname
	       << "clusterTOF=" << cacheTOF << "sectTOF=" << sectTOF << "plateTOF=" << plateTOF << "stripTOF=" << stripTOF << "padXTOF=" << padXTOF << "padZTOF=" << padZTOF
	       << "crossedStrip=" << crossedStrip << "sectPropagation=" << sectPropagation << "platePropagation=" << platePropagation << "stripPropagation=" << stripPropagation << "padXPropagation=" << padXPropagation
	       << "resX=" << resX << "resZ=" << resZ << "res=" << res << "track=" << trk
	       << "TPClabelTrackID=" << TPClabelTrackID << "TPClabelEventID=" << TPClabelEventID << "TPClabelSourceID=" << TPClabelSourceID
	       << "ITSlabelTrackID=" << ITSlabelTrackID << "ITSlabelEventID=" << ITSlabelEventID << "ITSlabelSourceID=" << ITSlabelSourceID
	       << "TOFlabelTrackID0=" << TOFlabelTrackID0 << "TOFlabelEventID0=" << TOFlabelEventID0 << "TOFlabelSourceID0=" << TOFlabelSourceID0
	       << "TOFlabelTrackID1=" << TOFlabelTrackID1 << "TOFlabelEventID1=" << TOFlabelEventID1 << "TOFlabelSourceID1=" << TOFlabelSourceID1
	       << "TOFlabelTrackID2=" << TOFlabelTrackID2 << "TOFlabelEventID2=" << TOFlabelEventID2 << "TOFlabelSourceID2=" << TOFlabelSourceID2
	       << "intLength=" << intLength << "intTimePion=" << intTimePion << "timeTOF=" << timeTOF
	       << "\n";
  }
  mTimerDBG.Stop();
}
*/
#endif
