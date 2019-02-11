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

  for(int ipad=0; ipad < NPADSPERSTEP; ipad++){
  }
  mHistoChTimeSlewingAll = new TH2F("hLHCchTimeSlewingAll", ";tot (ns);channel offset (ps)", 40, 0, 25, 200, -24400, 24400);

}
//______________________________________________
CalibTOF::~CalibTOF(){

  // destructor

  if (mHistoLHCphase) delete mHistoLHCphase;
  delete mHistoChTimeSlewingAll;
  
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
}

//______________________________________________
void CalibTOF::init()
{
  ///< initizalizations

  if (mInitDone) {
    LOG(ERROR) << "Initialization was already done";
    return;
  }

  TH1::AddDirectory(0); // needed because we have the LHCPhase created here, while in the macro we might have the output file open
                        // (we don't want to bind the histogram to the file, or teh destructor will complain)

  attachInputTrees();

  std::fill_n(mCalibChannelOffset, o2::tof::Geo::NCHANNELS, 0);
  std::fill_n(mCalibChannelOffsetErr, o2::tof::Geo::NCHANNELS, -1);
  std::fill_n(mInitialCalibChannelOffset, o2::tof::Geo::NCHANNELS, 0);
  
  // load better knoldge of channel offset (from CCDB?)
  // to be done

  // create output branch with output -- for now this is empty
  if (mOutputTree) {
    mOutputTree->Branch("LHCphaseMeasurementInterval", &mNLHCphaseIntervals, "LHCphaseMeasurementInterval/I");
    mOutputTree->Branch("LHCphase", mLHCphase, "LHCphase[LHCphaseMeasurementInterval]/F");
    mOutputTree->Branch("LHCphaseErr", mLHCphaseErr, "LHCphaseErr[LHCphaseMeasurementInterval]/F");
    mOutputTree->Branch("LHCphaseStartInterval", mLHCphaseStartInterval, "LHCphaseStartInterval[LHCphaseMeasurementInterval]/I");
    mOutputTree->Branch("LHCphaseEndInterval", mLHCphaseEndInterval, "LHCphaseEndInterval[LHCphaseMeasurementInterval]/I");
    mOutputTree->Branch("nChannels", &mNChannels, "nChannels/I");
    mOutputTree->Branch("ChannelOffset", mCalibChannelOffset, "ChannelOffset[nChannels]/F");
    mOutputTree->Branch("ChannelOffsetErr", mCalibChannelOffsetErr, "ChannelOffsetErr[nChannels]/F");
    //    LOG(INFO) << "Matched tracks will be stored in " << mOutputBranchName << " branch of tree "
    //              << mOutputTree->GetName();
  } else {
    LOG(ERROR) << "Output tree is not attached, matched tracks will not be stored";
  }

  // booking the histogram of the LHCphase
  int nbinsLHCphase = TMath::Min(1000, int((mMaxTimestamp - mMinTimestamp)/300)+1);
  if (nbinsLHCphase < 1000) mMaxTimestamp = mMinTimestamp + mNLHCphaseIntervals*300; // we want that the last bin of the histogram is also large 300s; this we need to do only when we have less than 1000 bins, because in this case we will integrate over intervals that are larger than 300s anyway
  mHistoLHCphase = new TH2F("hLHCphase", ";clock offset (ps); timestamp (s)", 1000, -24400, 24400, nbinsLHCphase, mMinTimestamp, mMaxTimestamp);

  mInitDone = true;

  mTimerTot.Stop();
  mTimerTot.Reset();

  print();
}
//______________________________________________
void CalibTOF::run(int flag, int sector)
{
  ///< running the matching

  TTree *localTree = mTreeCollectedCalibInfoTOF;
  TFile *fOpenLocally = nullptr;
  Int_t   currTOFInfoTreeEntry = -1;
  std::vector<o2::dataformats::CalibInfoTOFshort>* localCalibInfoTOF = mCalibInfoTOF;
  if(sector != -1){ // load tree as a new instance to read it in parallel with other processes
    fOpenLocally = TFile::Open(localTree->GetCurrentFile()->GetName());
    localTree = (TTree *) fOpenLocally->Get(localTree->GetName());
    localTree->SetBranchAddress(mCollectedCalibInfoTOFBranchName.data(), &localCalibInfoTOF);

    printf("nentries = %d\n",localTree->GetEntries());
  }


  if (!mInitDone) {
    LOG(FATAL) << "init() was not done yet";
  }

  //  mTimerTot.Start();

  if (flag & kLHCphase) { // LHC phase --> we will use all the entries in the tree
    while(loadTOFCollectedCalibInfo(localTree,currTOFInfoTreeEntry)){ // fill here all histos you need 
      fillLHCphaseCalibInput(localCalibInfoTOF); // we will fill the input for the LHC phase calibration
    }
    doLHCPhaseCalib();
  }
  //  else { // channel offset + problematic (flag = 1), or time slewing (flag = 2)
  if ((flag & kChannelOffset) || (flag & kChannelTimeSlewing)) { // for the moment compute everything idependetly of the flag
    TH1F* histoChOffsetTemp[NPADSPERSTEP];
    std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad[NPADSPERSTEP];
    for (int ipad = 0; ipad < NPADSPERSTEP; ipad++){
      histoChOffsetTemp[ipad] = new TH1F(Form("hLHCchOffsetTemp_%02d_%04d", sector, ipad), Form("Sector %02d;channel offset (ps)", ipad, sector), 1000, -24400, 24400);
      if (flag & kChannelTimeSlewing) calibTimePad[ipad] = new std::vector<o2::dataformats::CalibInfoTOFshort>; // temporary array containing [time, tot] for every pad that we process; this will be the input for the 2D histo for timeSlewing calibration (to be filled after we get the channel offset)
      else calibTimePad[ipad] = nullptr;
    }
    
    TF1* funcChOffset = new TF1(Form("fLHCchOffset_%02d", sector), "gaus");
    funcChOffset->SetParLimits(2, 100, 400);
    funcChOffset->SetParLimits(1, -12500, 12500);
    TH2F* histoChTimeSlewingTemp = new TH2F(Form("hLHCchTimeSlewingTemp_%02d", sector), Form("Sector %02d;tot (ns);channel offset (ps)", sector), 40, 0, 25, 200, -24400, 24400); 

    int startLoop = 0; // first pad that we will process in this process (we are processing a sector, unless sector = -1)
    int endLoop = o2::tof::Geo::NCHANNELS; // last pad that we will process in this process (we are processing a sector)
    if (sector > -1) {
      startLoop = sector*o2::tof::Geo::NPADSXSECTOR; // first pad that we will process in this process (we are processing a sector, unless sector = -1)
      endLoop = startLoop + o2::tof::Geo::NPADSXSECTOR; // last pad that we will process in this process (we are processing a sector, unless sector = -1)
    }
    for (int ich = startLoop; ich < endLoop; ich += NPADSPERSTEP){
      sector = ich/o2::tof::Geo::NPADSXSECTOR; // we change the value of sector which is needed when it is "-1" to put
                                               // in the output histograms a meaningful name; this is not needed in
                                               // case we run with sector != -1, but it will not hurt :) 
      resetChannelLevelHistos(flag, histoChOffsetTemp, histoChTimeSlewingTemp, calibTimePad);
      printf("strip %i\n", ich/96);
      currTOFInfoTreeEntry = ich - 1;
      int ipad = 0;
      int entryNext = currTOFInfoTreeEntry + o2::tof::Geo::NCHANNELS;

      while (loadTOFCollectedCalibInfo(localTree,currTOFInfoTreeEntry)) { // fill here all histos you need 

	fillChannelCalibInput(localCalibInfoTOF, mInitialCalibChannelOffset[ich+ipad], ipad, histoChOffsetTemp[ipad], calibTimePad[ipad]); // we will fill the input for the channel-level calibration
	ipad++;

	if(ipad == NPADSPERSTEP){
	  ipad = 0;
	  currTOFInfoTreeEntry = entryNext;
	  entryNext += o2::tof::Geo::NCHANNELS;
	}
      }
      TFile * fout = nullptr;
      if (flag & kChannelTimeSlewing) fout = new TFile(Form("timeslewingTOF%06i.root",ich/96),"RECREATE");
      
      for (ipad = 0; ipad < NPADSPERSTEP; ipad++){
	if (histoChOffsetTemp[ipad]->GetEntries() > 30){
	  doChannelLevelCalibration(flag, ipad, histoChOffsetTemp[ipad], funcChOffset);
	  mCalibChannelOffset[ich+ipad] = funcChOffset->GetParameter(1) + mInitialCalibChannelOffset[ich+ipad];
	  mCalibChannelOffsetErr[ich+ipad] = funcChOffset->GetParError(1);

	  // now fill 2D histo for time-slewing using current channel offset
	  
	  if (flag & kChannelTimeSlewing) {
	    histoChTimeSlewingTemp->Reset();
	    fillChannelTimeSlewingCalib(mCalibChannelOffset[ich+ipad], ipad, histoChTimeSlewingTemp, calibTimePad[ipad]); // we will fill the input for the channel-time-slewing calibration

	//	  histoChTimeSlewingTemp[ipad]->FitSlicesY(funcChOffset,0,-1,0,"R");

	    int ibin0 = 1;
	    int nbin = 0;
	    float xval[1000];
	    float val[1000];
	    float eval[1000];
	    for(int ibin = ibin0; ibin <= histoChTimeSlewingTemp->GetNbinsX(); ibin++){
	      if(ibin <  histoChTimeSlewingTemp->GetNbinsX()){ // if the integral of the next bins is lower than the threshold let's continue (to include also that entries in the last bin)
		TH1D *hLast = histoChTimeSlewingTemp->ProjectionY("tempProjTimeSlewingLast", ibin+1, histoChTimeSlewingTemp->GetNbinsX());
		if (hLast->GetEntries() < 50) {
		  delete hLast;
		  continue;
		}
		delete hLast;
	      }
	      TH1D* h = histoChTimeSlewingTemp->ProjectionY("tempProjTimeSlewingFit", ibin0, ibin);
	      if (h->GetEntries() < 50) {
		delete h;
		continue;
	      }
	      h->Fit(funcChOffset, "WW", "");
	      h->Fit(funcChOffset, "", "", funcChOffset->GetParameter(1)-600, funcChOffset->GetParameter(1)+400);
	      delete h;
	      xval[nbin] = histoChTimeSlewingTemp->GetXaxis()->GetBinCenter(ibin0) - histoChTimeSlewingTemp->GetXaxis()->GetBinWidth(ibin0)*0.5;
	      xval[nbin+1] = histoChTimeSlewingTemp->GetXaxis()->GetBinCenter(ibin) + histoChTimeSlewingTemp->GetXaxis()->GetBinWidth(ibin)*0.5;
	      val[nbin] = funcChOffset->GetParameter(1);
	      eval[nbin] = funcChOffset->GetParError(1);
	      nbin++;
	      ibin0 = ibin+1;
	    }
	  
	    if (nbin) {
	      int istrip = ((ich+ipad)/o2::tof::Geo::NPADS)%o2::tof::Geo::NSTRIPXSECTOR;
	      mProjTimeSlewingTemp = new TH1D(Form("pad_%02d_%02d_%02d", sector, istrip, ipad%o2::tof::Geo::NPADS), "", nbin, xval);
	      
	      for(int ibin=1 ;ibin <= nbin; ibin++){
		mProjTimeSlewingTemp->SetBinContent(ibin, val[ibin]);
	      	mProjTimeSlewingTemp->SetBinError(ibin, eval[ibin]);
	      }
	      mProjTimeSlewingTemp->Write();
	      histoChTimeSlewingTemp->Write(Form("histoChTimeSlewingTemp_%02d_%02d_%02d", sector, istrip, ipad%o2::tof::Geo::NPADS));
	    }
	  }
	}
      }
      if (fout) fout->Close();
    }

    for(int ipad=0; ipad < NPADSPERSTEP; ipad++){
      delete histoChOffsetTemp[ipad];
      if (calibTimePad[ipad]) delete calibTimePad[ipad];
    }
    delete histoChTimeSlewingTemp;
    delete funcChOffset;
  }

  if(fOpenLocally) fOpenLocally->Close();

  //  mTimerTot.Stop();
  //  printf("Timing:\n");
  //  printf("Total:        ");
  //  mTimerTot.Print();
}

//______________________________________________
void CalibTOF::fillOutput(){
  mOutputTree->Fill();
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
bool CalibTOF::loadTOFCollectedCalibInfo(TTree *localTree, int &currententry, int increment)
{
  ///< load next chunk of TOF infos
  //  printf("Loading TOF calib infos: number of entries in tree = %lld\n", mTreeCollectedCalibInfoTOF->GetEntries());

  currententry += increment;
  //while (currententry < localTree->GetEntries()){
  while (currententry < 800000){
	 //    && currententry < o2::tof::Geo::NCHANNELS) {
    localTree->GetEntry(currententry);
    //LOG(INFO) << "Loading TOF calib info entry " << currententry << " -> " << mCalibInfoTOF->size()<< " infos";
        
    return true;
  }
  currententry -= increment;

  return false;
}

//______________________________________________

void CalibTOF::fillLHCphaseCalibInput(std::vector<o2::dataformats::CalibInfoTOFshort>* calibinfotof){
  
  // we will fill the input for the LHC phase calibration
  
  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1./bc;
    
  // implemented for flag=0, channel=-1 (-1 means all!)
  for(auto infotof = calibinfotof->begin(); infotof != calibinfotof->end(); infotof++){
    double dtime = infotof->getDeltaTimePi();
    dtime -= int(dtime*bc_inv + 0.5)*bc;
    
    mHistoLHCphase->Fill(dtime, infotof->getTimestamp());
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
  int ifit0 = 1;
  for (int ifit = ifit0; ifit <= mHistoLHCphase->GetNbinsY(); ifit++){
    TH1D* htemp = mHistoLHCphase->ProjectionX("htemp", ifit0, ifit);
    if (htemp->GetEntries() < 100) {
      // we cannot fit the histogram, we will merge with the next bin
      Printf("We don't have enough entries to fit");
      continue;
    }
    htemp->Fit(mFuncLHCphase, "WW", "Q");
    htemp->Fit(mFuncLHCphase, "", "Q", mFuncLHCphase->GetParameter(1)-600, mFuncLHCphase->GetParameter(1)+400);
    // TODO: check that the fit really worked before filling all below
    mLHCphase[mNLHCphaseIntervals] = mFuncLHCphase->GetParameter(1);
    mLHCphaseErr[mNLHCphaseIntervals] = mFuncLHCphase->GetParError(1);
    mLHCphaseStartInterval[mNLHCphaseIntervals] = mHistoLHCphase->GetYaxis()->GetBinLowEdge(ifit0); // from when the interval 
    mLHCphaseEndInterval[mNLHCphaseIntervals] = mHistoLHCphase->GetYaxis()->GetBinUpEdge(ifit);
    ifit0 = ifit+1; // starting point for the next LHC interval
    mNLHCphaseIntervals++; // how many intervals we have calibrated so far
  }
}
//______________________________________________

void CalibTOF::fillChannelCalibInput(std::vector<o2::dataformats::CalibInfoTOFshort>* calibinfotof, float offset, int ipad, TH1F* histo, std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad){
  
  // we will fill the input for the channel-level calibration

  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1./bc;
    
  // implemented for flag=0, channel=-1 (-1 means all!)
  for(auto infotof = calibinfotof->begin(); infotof != calibinfotof->end(); infotof++){
    double dtime = infotof->getDeltaTimePi() - offset; // removing existing offset 
    dtime -= int(dtime*bc_inv + 0.5)*bc;
    
    histo->Fill(dtime);
    if (calibTimePad) calibTimePad->push_back(*infotof);
  }
}
//______________________________________________

void CalibTOF::fillChannelTimeSlewingCalib(float offset, int ipad, TH2F* histo, std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad){
  
// we will fill the input for the channel-time-slewing calibration

  static double bc = 1.e13 / o2::constants::lhc::LHCRFFreq; // bunch crossing period (ps)
  static double bc_inv = 1./bc;
    
  // implemented for flag=0, channel=-1 (-1 means all!)
  for(auto infotof = calibTimePad->begin(); infotof != calibTimePad->end(); infotof++){
    double dtime = infotof->getDeltaTimePi() - offset; // removing the already calculated offset; this is needed to
                                                       // fill the time slewing histogram in the correct range 
    dtime -= int(dtime*bc_inv + 0.5)*bc;
    
    histo->Fill(TMath::Min(double(infotof->getTot()), 24.9), dtime);
    mHistoChTimeSlewingAll->Fill(infotof->getTot(), dtime);
  }
}
//______________________________________________

void CalibTOF::doChannelLevelCalibration(int flag, int ipad, TH1F* histo, TF1* funcChOffset){

  // calibrate single channel from histos - offsets

  funcChOffset->SetParameter(0, 100);
  funcChOffset->SetParameter(1, histo->GetMean());
  funcChOffset->SetParameter(2, 200);

  histo->Fit(funcChOffset, "WW", "Q"); 
  histo->Fit(funcChOffset, "", "Q", funcChOffset->GetParameter(1)-600, funcChOffset->GetParameter(1)+400); 

}
//______________________________________________

void CalibTOF::resetChannelLevelHistos(int flag, TH1F* histoOffset[NPADSPERSTEP], TH2F* histoTimeSlewing, std::vector<o2::dataformats::CalibInfoTOFshort>* calibTimePad[NPADSPERSTEP]){
  
  // reset signle channel histos

  for(int ipad=0; ipad < NPADSPERSTEP; ipad++){
    histoOffset[ipad]->Reset();
    if (calibTimePad[ipad]) calibTimePad[ipad]->clear();
  }
  histoTimeSlewing->Reset();
}

