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

#include "TMath.h"

using namespace o2::globaltracking;

ClassImp(CalibTOF);

//______________________________________________
CalibTOF::CalibTOF(){

  // constructor needed to instantiate the pointers of the class (histos + array)

  mHistoLHCphase = new TH1F("hLHCphase", ";clock offset (ps)", 1000, -24400, 24400);
  for(int ipad=0; ipad < NPADSPERSTEP; ipad++){
    mHistoChOffsetTemp[ipad]  = new TH1F(Form("hLHCchOffsetTemp%i", ipad), ";channel offset (ps)", 1000, -24400, 24400);
    mHistoChTimeSlewingTemp[ipad]  = new TH2F(Form("hLHCchTimeSlewingTemp%i", ipad), ";tot (ns);channel offset (ps)", 40, 0, 25, 200, -24400, 24400);
    mCalibTimePad[ipad] = new std::vector<o2::dataformats::CalibInfoTOFshort>; 
  }
  mHistoChTimeSlewingAll = new TH2F("hLHCchTimeSlewingAll", ";tot (ns);channel offset (ps)", 40, 0, 25, 200, -24400, 24400);

}
//______________________________________________
CalibTOF::~CalibTOF(){

  // destructor

  delete mHistoLHCphase;
  for(int ipad=0; ipad < NPADSPERSTEP; ipad++){
    delete mHistoChOffsetTemp[ipad];
    delete mHistoChTimeSlewingTemp[ipad];
    delete mCalibTimePad[ipad];
  }
  delete mHistoChTimeSlewingAll;
  
}  
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
  if (flag == 0) { // for the moment compute everything idependetly of the flag
    for (int ich = 0; ich < o2::tof::Geo::NCHANNELS; ich += NPADSPERSTEP){
      resetChannelLevelHistos(flag);
      printf("strip %i\n", ich/96);
      mCurrTOFInfoTreeEntry = ich - 1;
      int ipad = 0;
      int entryNext = mCurrTOFInfoTreeEntry + o2::tof::Geo::NCHANNELS;

      while (loadTOFCollectedCalibInfo()) { // fill here all histos you need 

	fillChannelCalibInput(mInitialCalibChannelOffset[ich+ipad], ipad); // we will fill the input for the channel-level calibration
	ipad++;

	if(ipad == NPADSPERSTEP){
	  ipad = 0;
	  mCurrTOFInfoTreeEntry = entryNext;
	  entryNext += o2::tof::Geo::NCHANNELS;
	}
      }

      TFile fout(Form("timeslewingTOF%06i.root",ich/96),"RECREATE");

      for (ipad = 0; ipad < NPADSPERSTEP; ipad++){
	if (mHistoChOffsetTemp[ipad]->GetEntries() > 30){
	  doChannelLevelCalibration(flag, ipad);
	  mCalibChannelOffset[ich+ipad] = mFuncChOffset->GetParameter(1) + mInitialCalibChannelOffset[ich+ipad];
	  mCalibChannelOffsetErr[ich+ipad] = mFuncChOffset->GetParError(1);

	  // now fill 2D histo for time-slewing using current channel offset
	  fillChannelTimeSlewingCalib(mCalibChannelOffset[ich+ipad], ipad); // we will fill the input for the channel-time-slewing calibration

	//	  mHistoChTimeSlewingTemp[ipad]->FitSlicesY(mFuncChOffset,0,-1,0,"R");

	  int ibin0 = 1;
	  int nbin = 0;
	  float xval[1000];
	  float val[1000];
	  float eval[1000];
	  for(int ibin = ibin0; ibin <= mHistoChTimeSlewingTemp[ipad]->GetNbinsX(); ibin++){
	    if(ibin <  mHistoChTimeSlewingTemp[ipad]->GetNbinsX()){ // if the integral of the next bins is lower than the threshold let's continue (to include also that entries in the last bin)
	      TH1D *hLast = mHistoChTimeSlewingTemp[ipad]->ProjectionY("tempProjTimeSlewingLast", ibin+1, mHistoChTimeSlewingTemp[ipad]->GetNbinsX());
	      if (hLast->GetEntries() < 50) continue;
	    }
	    TH1D *h = mHistoChTimeSlewingTemp[ipad]->ProjectionY("tempProjTimeSlewingFit", ibin0, ibin);
	    if (h->GetEntries() < 50) continue;
	    Printf("Fitting bin %d of the time slewing 2D distribution - with WW", ibin);
	    h->Fit(mFuncChOffset, "WW", "");
	    Printf("Fitting bin %d of the time slewing 2D distribution - without WW", ibin);
	    h->Fit(mFuncChOffset, "", "", mFuncChOffset->GetParameter(1)-600, mFuncChOffset->GetParameter(1)+400);
	    //	    h->Fit(mFuncChOffset, "", "", h->GetMean()-600, h->GetMean()+400);
	    printf("\n%i) value = %f %f\n", ibin, mFuncChOffset->GetParameter(1), mFuncChOffset->GetParError(1));
	    xval[nbin] = mHistoChTimeSlewingTemp[ipad]->GetXaxis()->GetBinCenter(ibin0) - mHistoChTimeSlewingTemp[ipad]->GetXaxis()->GetBinWidth(ibin0)*0.5;
	    xval[nbin+1] = mHistoChTimeSlewingTemp[ipad]->GetXaxis()->GetBinCenter(ibin) + mHistoChTimeSlewingTemp[ipad]->GetXaxis()->GetBinWidth(ibin)*0.5;
	    val[nbin] = mFuncChOffset->GetParameter(1);
	    eval[nbin] = mFuncChOffset->GetParError(1);
	    nbin++;
	    ibin0 = ibin+1;
	  }
	  
	  if (nbin) {
	    mProjTimeSlewingTemp = new TH1D(Form("pad%02i", ipad), "", nbin, xval);
	    
	    for(int ibin=1 ;ibin <= nbin; ibin++){
	      mProjTimeSlewingTemp->SetBinContent(ibin, val[ibin]);
	      mProjTimeSlewingTemp->SetBinError(ibin, eval[ibin]);
	    }
	    mProjTimeSlewingTemp->Write();
	    mHistoChTimeSlewingTemp[ipad]->Write();
	  }
	}
      }
      fout.Close();
      
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
  std::fill_n(mCalibChannelOffsetErr, o2::tof::Geo::NCHANNELS, -1);
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

  mTimerTot.Stop();
  mTimerTot.Reset();

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
  while (mCurrTOFInfoTreeEntry < mTreeCollectedCalibInfoTOF->GetEntries()){
	 //    && mCurrTOFInfoTreeEntry < o2::tof::Geo::NCHANNELS) {
    mTreeCollectedCalibInfoTOF->GetEntry(mCurrTOFInfoTreeEntry);
    //LOG(INFO) << "Loading TOF calib info entry " << mCurrTOFInfoTreeEntry << " -> " << mCalibInfoTOF->size()<< " infos";
        
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

  if (!mFuncLHCphase){
    mFuncLHCphase = new TF1("fLHCphase", "gaus");
    mFuncLHCphase->SetParameter(0, 1000);
    mFuncLHCphase->SetParameter(1, 0);
    mFuncLHCphase->SetParameter(2, 200);
    mFuncLHCphase->SetParLimits(2, 100, 400);
  }
  Printf("Fitting LHC phase with WW");
  mHistoLHCphase->Fit(mFuncLHCphase, "WW", "Q");
  Printf("Fitting LHC phase without WW");
  mHistoLHCphase->Fit(mFuncLHCphase, "", "Q", mFuncLHCphase->GetParameter(1)-600, mFuncLHCphase->GetParameter(1)+400);
}
//______________________________________________

void CalibTOF::fillChannelCalibInput(float offset, int ipad){
  
  // we will fill the input for the channel-level calibration

  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1./bc;
    
  // implemented for flag=0, channel=-1 (-1 means all!)
  for(auto infotof = mCalibInfoTOF->begin(); infotof != mCalibInfoTOF->end(); infotof++){
    double dtime = infotof->getDeltaTimePi() - offset; // removing existing offset 
    dtime -= int(dtime*bc_inv + 0.5)*bc;
    
    mHistoChOffsetTemp[ipad]->Fill(dtime);
    mCalibTimePad[ipad]->push_back(*infotof);
  }
}
//______________________________________________

void CalibTOF::fillChannelTimeSlewingCalib(float offset, int ipad){
  
// we will fill the input for the channel-time-slewing calibration

  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1./bc;
    
  // implemented for flag=0, channel=-1 (-1 means all!)
  for(auto infotof = mCalibTimePad[ipad]->begin(); infotof != mCalibTimePad[ipad]->end(); infotof++){
    double dtime = infotof->getDeltaTimePi() - offset; // removing the already calculated offset; this is needed to
                                                       // fill the time slewing histogram in the correct range 
    dtime -= int(dtime*bc_inv + 0.5)*bc;
    
    mHistoChTimeSlewingTemp[ipad]->Fill(TMath::Min(double(infotof->getTot()), 24.9), dtime);
    mHistoChTimeSlewingAll->Fill(infotof->getTot(), dtime);
  }
}
//______________________________________________

void CalibTOF::doChannelLevelCalibration(int flag, int ipad){

  // calibrate single channel from histos - offsets

  if(!mFuncChOffset){
    mFuncChOffset = new TF1("fLHCchOffset", "gaus");
    mFuncChOffset->SetParLimits(2, 100, 400);
    mFuncChOffset->SetParLimits(1, -12500, 12500);
  }
  mFuncChOffset->SetParameter(0, 100);
  mFuncChOffset->SetParameter(1, mHistoChOffsetTemp[ipad]->GetMean());
  mFuncChOffset->SetParameter(2, 200);

  Printf("First fitting with option WW for offset");
  mHistoChOffsetTemp[ipad]->Fit(mFuncChOffset, "WW", "Q"); 
  Printf("Second fitting without option WW for offset");
  mHistoChOffsetTemp[ipad]->Fit(mFuncChOffset, "", "Q", mFuncChOffset->GetParameter(1)-600, mFuncChOffset->GetParameter(1)+400); 

}
//______________________________________________

void CalibTOF::resetChannelLevelHistos(int flag){
  
  // reset signle channel histos

  for(int ipad=0; ipad < NPADSPERSTEP; ipad++){
    mHistoChOffsetTemp[ipad]->Reset();
    mHistoChTimeSlewingTemp[ipad]->Reset();
    mCalibTimePad[ipad]->clear();
  }
}

