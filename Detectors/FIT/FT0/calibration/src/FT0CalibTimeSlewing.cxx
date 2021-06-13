// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \file FT0CalibTimeSlewing.cxx
/// \brief Class for  slewing calibration object
///
#include <algorithm>
#include <cstdio>
#include <TH1F.h>
#include <TF1.h>
#include <TFitResult.h>
#include <TFileMerger.h>
#include <TFile.h>
#include "FT0Calibration/FT0CalibTimeSlewing.h"

using namespace o2::ft0;

FT0CalibTimeSlewing::FT0CalibTimeSlewing()
{
  for (int iCh = 0; iCh < NCHANNELS; iCh++) {
    mSigmaPeak[iCh] = -1.;
    mTimeAmpHist[iCh] = new TH2F(Form("hTimeAmpHist%d", iCh), Form("TimeAmp%d", iCh),
                                 NUMBER_OF_HISTOGRAM_BINS_X, 0, HISTOGRAM_RANGE_X,
                                 NUMBER_OF_HISTOGRAM_BINS_Y, -HISTOGRAM_RANGE_Y, HISTOGRAM_RANGE_Y);
  }
}

//______________________________________________
float FT0CalibTimeSlewing::getChannelOffset(int channel, int amplitude) const
{
  return mTimeSlewing[channel].Eval(amplitude);
}

//______________________________________________
void FT0CalibTimeSlewing::fillGraph(int channel, TH2F* histo)
{
  LOG(INFO) << "FT0CalibTimeSlewing::fillGraph " << channel << " entries " << int(histo->GetEntries());
  double shiftchannel = 0;
  TH1D* hist_Proj = histo->ProjectionY();
  TFitResultPtr res = hist_Proj->Fit("gaus", "SQ");
  if ((Int_t)res == 0) {
    shiftchannel = res->Parameter(1);
  }
  Double_t xgr[NUMBER_OF_HISTOGRAM_BINS_X] = {};
  Double_t ygr[NUMBER_OF_HISTOGRAM_BINS_X] = {};
  TH1D* proj = nullptr;
  int nbins = 0;
  for (int ibin = 1; ibin < NUMBER_OF_HISTOGRAM_BINS_X; ibin++) {
    xgr[ibin] = histo->GetXaxis()->GetBinCenter(ibin);
    proj = histo->ProjectionY(Form("proj_px%i", ibin), ibin, ibin + 1);
    if (proj->GetEntries() < 500) {
      ygr[ibin] = 0;
      continue;
    }
    TFitResultPtr r = proj->Fit("gaus", "SQ");
    if ((Int_t)r == 0) {
      ygr[ibin] = r->Parameter(1) - shiftchannel;
    }
    nbins++;
    LOG(INFO) << "channel " << channel << " bin " << ibin << " x " << xgr[ibin] << " y " << ygr[ibin] << " ent " << proj->GetEntries() << " sigma " << r->Parameter(2) << " shiftchannel " << shiftchannel;
  }
  TGraph* grTimeAmp = new TGraph(nbins + 5, xgr, ygr);
  mTimeSlewing[channel] = *grTimeAmp;
}
//______________________________________________
FT0CalibTimeSlewing& FT0CalibTimeSlewing::operator+=(const FT0CalibTimeSlewing& other)
{
  for (int i = 0; i < NCHANNELS; i++) {
    mTimeSlewing[i] = other.mTimeSlewing[i];
    mSigmaPeak[i] = other.mSigmaPeak[i];
  }
  return *this;
}

//______________________________________________
void FT0CalibTimeSlewing::mergeFilesWithTree()
{
  TFileMerger merger;
  merger.OutputFile(mMergedFileName.c_str());
  for (Int_t i = 0; i < mNfiles; i++) {
    TFile* file =
      TFile::Open(Form("%s_%d.root", mSingleFileName.c_str(), i));
    if (file) {
      merger.AddAdoptFile(file);
    }
  }
  if (!merger.Merge()) {
    LOG(FATAL) << "Could not merge files";
  }
  TFile mMergedFile{merger.GetOutputFileName()};
  TTree* tr = (TTree*)mMergedFile.Get("treeCollectedCalibInfo");
  if (!tr) {
    LOG(FATAL) << "Could not get tree with calib info";
  }
  fillHistos(tr);
}

//______________________________________________
void FT0CalibTimeSlewing::fillHistos(TTree* tr)
{
  std::vector<o2::ft0::FT0CalibrationInfoObject>* localCalibInfoFT0 = nullptr;
  if (!tr->GetBranch("FT0CollectedCalibInfo")) {
    LOG(FATAL) << "Did not find collected FT0 calib info branch  in the input tree";
  }
  tr->SetBranchAddress("FT0CollectedCalibInfo", &localCalibInfoFT0);
  for (Int_t ievent = 0; ievent < tr->GetEntries(); ievent++) {
    tr->GetEvent(ievent);
    for (auto& info : *localCalibInfoFT0) {
      LOG(DEBUG) << " ch " << int(info.getChannelIndex()) << " time " << info.getTime() << " amp " << info.getAmp();
      int iCh = info.getChannelIndex();
      mTimeAmpHist[iCh]->Fill(info.getAmp(), info.getTime());
    }
  }
  delete localCalibInfoFT0;
}
