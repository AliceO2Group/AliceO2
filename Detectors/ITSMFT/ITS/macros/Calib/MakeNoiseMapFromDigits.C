#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/Digit.h"

#pragma link C++ class std::vector < std::map < int, int>> + ;

void MakeNoiseMapFromDigits(std::string digifile = "itsdigits.root", int hitCut = 3)
{
  // Digits
  TFile* file1 = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)file1->Get("o2sim");
  std::vector<o2::itsmft::Digit>* digArr = nullptr;
  digTree->SetBranchAddress("ITSDigit", &digArr);

  std::vector<std::map<int, int>> noisyPixels;
  noisyPixels.assign(24120, std::map<int, int>());

  int nevD = digTree->GetEntries(), nd = 0;
  for (int iev = 0; iev < nevD; iev++) {
    digTree->GetEvent(iev);
    for (const auto& d : *digArr) {
      nd++;
      auto id = d.getChipIndex();
      auto row = d.getRow();
      auto col = d.getColumn();
      auto key = row * 1024 + col;
      noisyPixels[id][key]++;
    }
  }

  int nPixelCalib = 0;
  for (int i = 0; i < noisyPixels.size(); i++) {
    const auto& map = noisyPixels[i];
    for (const auto& pair : map) {
      if (pair.second > hitCut) {
        auto key = pair.first;
        auto row = key / 1024;
        auto col = key % 1024;
        std::cout << "Chip, row, col: " << i << ' ' << row << ' ' << col << "  Hits: " << pair.second << '\n';
        nPixelCalib++;
      }
    }
  }

  TFile* fout = new TFile("ITSnoise.root", "new");
  fout->cd();
  fout->WriteObject(&noisyPixels, "Noise");
  fout->Close();

  std::cout << "Total Digits Processed = " << nd << '\n';
  std::cout << "Noise threshold = " << hitCut << "  Noisy pixels = " << nPixelCalib << '\n';
}
