#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "TLeaf.h"
#include <stdio.h>
#include "DataFormatsTOF/CalibInfoTOF.h"
#endif

void convertTreeTo02object()
{

  TFile* fout = TFile::Open("TOFcalibTreeOut.root", "RECREATE");
  TTree* tout = new TTree("calibTOF", "Calib TOF infos");

  TFile* f = TFile::Open("TOFcalibTree.root");
  TTree* t = (TTree*)f->Get("aodTree");
  std::vector<o2::dataformats::CalibInfoTOF> mCalibInfoTOF;
  tout->Branch("TOFCalibInfo", &mCalibInfoTOF);

  Printf("The tree has %lld entries", t->GetEntries());
  for (int ientry = 1; ientry <= t->GetEntries(); ientry++) {
    if (ientry % 100 == 0)
      Printf("processing event %d", ientry);
    //for (int ientry = 1; ientry <= 3; ientry++){
    t->GetEntry(ientry);
    for (int i = 0; i < t->GetLeaf("nhits")->GetValue(); i++) {
      mCalibInfoTOF.emplace_back(t->GetLeaf("index")->GetValue(i), t->GetLeaf("timestamp")->GetValue(), t->GetLeaf("time")->GetValue(i) - t->GetLeaf("texp")->GetValue(i), t->GetLeaf("tot")->GetValue(i));
    }
    tout->Fill();
    mCalibInfoTOF.clear();
  }

  fout->cd();
  tout->Write();
}
