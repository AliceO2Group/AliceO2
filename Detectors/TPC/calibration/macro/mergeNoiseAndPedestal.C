// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TSystem.h"
#include "TFile.h"
#include "TPCBase/CalDet.h"
#include "TString.h"
#endif

void mergeNoiseAndPedestal(const std::string inFiles = "Event*.root", const std::string outFile = "mergedPedestalsAndNoise.root")
{
  TString files = gSystem->GetFromPipe(Form("ls %s", inFiles.data()));
  auto* arrFiles = files.Tokenize("\n");

  using namespace o2::tpc;

  CalDet<float> mergedPedestals("Pedestals");
  CalDet<float> mergedNoise("Noise");

  CalDet<float>* loadedPedestals = nullptr;
  CalDet<float>* loadedNoise = nullptr;

  for (auto* oFile : *arrFiles) {
    std::unique_ptr<TFile> f(TFile::Open(oFile->GetName()));
    f->GetObject("Pedestals", loadedPedestals);
    f->GetObject("Noise", loadedNoise);

    std::cout << "Merging file " << oFile->GetName() << "\n";

    mergedPedestals += (*loadedPedestals);
    mergedNoise += (*loadedNoise);
  }

  const auto entries = float(arrFiles->GetEntriesFast());
  if (entries > 0) {
    mergedPedestals /= entries;
    mergedNoise /= entries;
  }

  std::unique_ptr<TFile> fOut(TFile::Open(outFile.data(), "recreate"));
  fOut->WriteObject(&mergedPedestals, "Pedestals");
  fOut->WriteObject(&mergedNoise, "Noise");
}
