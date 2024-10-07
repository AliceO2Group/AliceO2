
#include "TFile.h"
#include "TNtuple.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TH1.h"
#include "TCanvas.h"

int32_t fastTransformQA()
{

  const char* fname = "fastTransformQA.root";

  TFile* file = new TFile(fname, "READ");

  if (!file || !file->IsOpen()) {
    printf("Can't open [%s] !\n", fname);
    return 1;
  }

  TNtuple* nt = (TNtuple*)file->FindObjectAny("fastTransformQA");
  if (!nt) {
    printf("Error getting TNuple fastTransformQA \n");
    return 1;
  }

  float sec, row, pad, time, x, y, z, fx, fy, fz;

  nt->SetBranchAddress("sec", &sec);
  nt->SetBranchAddress("row", &row);
  nt->SetBranchAddress("pad", &pad);
  nt->SetBranchAddress("time", &time);
  nt->SetBranchAddress("x", &x);
  nt->SetBranchAddress("y", &y);
  nt->SetBranchAddress("z", &z);
  nt->SetBranchAddress("fx", &fx);
  nt->SetBranchAddress("fy", &fy);
  nt->SetBranchAddress("fz", &fz);

  TCanvas* canv = new TCanvas("cfastTransform", "TPC fast transform QA", 2000, 500);
  canv->Draw();
  canv->Divide(3, 1);
  canv->Update();

  TH1F* qaX = new TH1F("qaX", "qaX [um]", 1000, -500., 500.);
  TH1F* qaY = new TH1F("qaY", "qaY [um]", 1000, -500., 500.);
  TH1F* qaZ = new TH1F("qaZ", "qaZ [um]", 1000, -500., 500.);

  for (int32_t i = 0; i < nt->GetEntriesFast(); i++) {
    if (i % 10000000 == 0) {
      std::cout << "processing " << i << " out of " << nt->GetEntriesFast() << "  (" << ((int64_t)i) * 100 / nt->GetEntriesFast() << " %)" << std::endl;
      canv->cd(1);
      qaX->Draw();
      canv->cd(2);
      qaY->Draw();
      canv->cd(3);
      qaZ->Draw();
      canv->cd(0);
      canv->Update();
    }
    int32_t ret = nt->GetEntry(i);
    if (ret <= 0) {
      std::cout << "Wrong entry, ret == " << ret << std::endl;
      continue;
    }
    qaX->Fill(1.e4 * (fx - x));
    qaY->Fill(1.e4 * (fy - y));
    qaZ->Fill(1.e4 * (fz - z));
  }

  canv->cd(1);
  qaX->Draw();
  canv->cd(2);
  qaY->Draw();
  canv->cd(3);
  qaZ->Draw();
  canv->cd(0);
  canv->Update();

  return 0;
}
