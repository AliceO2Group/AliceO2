#include "CalibTimeSlewingParamTOF.h"
#include "AliTOFCalibFineSlewing.h"
#include "TFile.h"
#include "TROOT.h"
#include "AliCDBEntry.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TTree.h"
#include "TMath.h"
#include "AliTOFChannelOffline.h"
#include "TH1C.h"
#include "AliTOFGeometry.h"

// macro to be run in *** AliRoot *** to convert the TOF OCDB entries in CCDB entries
// How to run (see below for futher instructions):
// gROOT->LoadMacro("CalibTimeSlewingParamTOF.cxx+")
// .x ConvertRun2CalibrationToO2.C+

class MyFineTimeSlewing : public AliTOFCalibFineSlewing

// class needed to access some data members of the AliTOFCalibFineSlewing class that are protected

{
 public:
  Int_t GetSize() { return fSize; }
  void GetChannelArrays(Int_t ich, Float_t* x, Float_t* y, Int_t& n)
  {
    n = fStart[ich + 1] - fStart[ich];
    if (ich == 157247)
      n = fSize - fStart[ich];
    for (Int_t i = 0; i < n; i++) {
      x[i] = fX[fStart[ich] + i] * 0.001; // in the OCDB, we saves the tot in ps
      y[i] = fY[fStart[ich] + i] * 1.;    // make it float
    }
  }

  ClassDef(MyFineTimeSlewing, 1);
};

void ConvertRun2CalibrationToO2()
{

  // Remember: Use AliRoot!!

  // actually you need to call this outside the macro, or it won't work

  gROOT->LoadMacro("CalibTimeSlewingParamTOF.cxx+");

  // so you should first:
  // - copy
  //      - DataFormats/Detectors/TOF/src/CalibTimeSlewingParamTOF.cxx
  //   and
  //      - DataFormats/Detectors/TOF/include/DataFormatsTOF/CalibTimeSlewingParamTOF.h
  //   in the local working directory, substituting:
  //      - #include "DataFormatsTOF/CalibTimeSlewingParamTOF.h"
  //   with
  //      - #include "CalibTimeSlewingParamTOF.h"
  // - then call:
  //      - gROOT->LoadMacro("CalibTimeSlewingParamTOF.cxx+")
  //   from the prompt

  o2::dataformats::CalibTimeSlewingParamTOF* mTimeSlewingObj = new o2::dataformats::CalibTimeSlewingParamTOF();

  TFile* ffineSlewing = new TFile("TOF/Calib/FineSlewing/Run0_999999999_v2_s0.root");
  AliCDBEntry* efineSlewing = (AliCDBEntry*)ffineSlewing->Get("AliCDBEntry");
  AliTOFCalibFineSlewing* fs = (AliTOFCalibFineSlewing*)efineSlewing->GetObject();
  TFile* foffset = new TFile("TOF/Calib/ParOffline/Run297624_999999999_v4_s0.root");
  AliCDBEntry* eoffset = (AliCDBEntry*)foffset->Get("AliCDBEntry");
  TObjArray* foff = (TObjArray*)eoffset->GetObject();
  TFile* fproblematic = new TFile("TOF/Calib/Problematic/Run296631_999999999_v3_s0.root");
  AliCDBEntry* eproblematic = (AliCDBEntry*)fproblematic->Get("AliCDBEntry");
  TH1C* hProb = (TH1C*)eproblematic->GetObject();

  MyFineTimeSlewing* mfs = (MyFineTimeSlewing*)fs;
  Printf("size = %d", mfs->GetSize());

  Float_t x[10000];
  Float_t y[10000];

  Int_t n;

  for (Int_t i = 0; i < 157248; i++) {
    mfs->GetChannelArrays(i, x, y, n);
    //Printf("channel %d has %d entries", i, n);
    AliTOFChannelOffline* parOffline = (AliTOFChannelOffline*)foff->At(i);
    //Printf("channel %d has offset = %f", parOffline->GetSlewPar(0));
    float corr0 = 0;
    for (Int_t islew = 0; islew < 6; islew++)
      corr0 += parOffline->GetSlewPar(islew) * TMath::Power(AliTOFGeometry::SlewTOTMin(), islew) * 1.e3;
    for (Int_t j = 0; j < n; j++) {
      float toteff = x[j];
      if (toteff < AliTOFGeometry::SlewTOTMin())
        toteff = AliTOFGeometry::SlewTOTMin();
      if (toteff > AliTOFGeometry::SlewTOTMax())
        toteff = AliTOFGeometry::SlewTOTMax();

      Float_t corr = 0;
      for (Int_t islew = 0; islew < 6; islew++)
        corr += parOffline->GetSlewPar(islew) * TMath::Power(toteff, islew) * 1.e3;
      if (j == 0 && x[j] > 0.03)
        mTimeSlewingObj->addTimeSlewingInfo(i, 0.025, y[j] + corr0); // force to have an entry for ToT=0
      mTimeSlewingObj->addTimeSlewingInfo(i, x[j], y[j] + corr);
    }
    if (n == 0) // force to have at least one entry
      mTimeSlewingObj->addTimeSlewingInfo(i, 0.025, corr0);

    // set problematics
    int sector = i / 8736;
    int localchannel = i % 8736;
    float fraction = float(hProb->GetBinContent(i) == 0) - 0.01; // negative means problematic
    mTimeSlewingObj->setFractionUnderPeak(sector, localchannel, fraction);
    mTimeSlewingObj->setSigmaPeak(sector, localchannel, 100.);
  }

  TGraph* g = new TGraph(n, x, y);

  new TCanvas();
  g->Draw("A*");

  Printf("time slewing object has size = %d", mTimeSlewingObj->size());

  TFile* fout = new TFile("outputCCDBfromOCDB.root", "RECREATE");
  TTree* t = new TTree("tree", "tree");
  t->Branch("CalibTimeSlewingParamTOF", &mTimeSlewingObj);
  t->Fill();
  t->Write();
  fout->Close();
}
