#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/NoiseMap.h"

void MakeNoiseMapFromDigits(std::string digifile = "itsdigits.root", int hitCut = 3)
{
  // Digits
  TFile* file1 = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)file1->Get("o2sim");
  std::vector<o2::itsmft::Digit>* digArr = nullptr;
  digTree->SetBranchAddress("ITSDigit", &digArr);

  o2::itsmft::NoiseMap noiseMap(24120);

  int nevD = digTree->GetEntries(), nd = 0;
  for (int iev = 0; iev < nevD; iev++) {
    digTree->GetEvent(iev);
    for (const auto& d : *digArr) {
      nd++;
      auto id = d.getChipIndex();
      auto row = d.getRow();
      auto col = d.getColumn();
      noiseMap.increaseNoiseCount(id, row, col);
    }
  }

  TFile* fout = new TFile("ITSnoise.root", "new");
  fout->cd();
  fout->WriteObject(&noiseMap, "Noise");
  fout->Close();

  int nPixelCalib = noiseMap.dumpAboveThreshold(hitCut);
  std::cout << "Noise threshold = " << hitCut << "  Noisy pixels = " << nPixelCalib << '\n';
  std::cout << "Total Digits Processed = " << nd << '\n';
}
