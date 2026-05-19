// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <string>
#include <string_view>
#include <vector>
#include <fmt/format.h>

#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"

#include "TPCBase/Utils.h"
#include "TPCCalibration/CMVContainer.h"
#include "TPCCalibration/CMVHelper.h"

#endif

using namespace o2::tpc;

/// Draw CMV (Common Mode Values) vs timebin from a CCDB TTree file
/// \param filename  input ROOT file containing the ccdb_object TTree
/// \param outDir    output directory for saved plots; nothing is saved if empty
/// \return          array of canvases
TObjArray* drawCMV(std::string_view filename, std::string_view outDir, std::string_view rootFileName = "CMVCanvases.root")
{
  TObjArray* arrCanvases = new TObjArray;
  arrCanvases->SetName("CMV");

  // open file
  CMVFileHandle fh;
  if (!fh.open(std::string(filename))) {
    fmt::print("ERROR: cannot open '{}'\n", filename);
    return arrCanvases;
  }
  fmt::print("Opened file: {}\n", filename);
  fmt::print("Tree 'ccdb_object' found, entries: {}\n", fh.tree->GetEntries());

  fmt::print("firstTF: {}, lastTF: {}\n", fh.firstTFInTree, fh.lastTFInTree);

  const int nEntries = fh.tree->GetEntries();
  if (nEntries == 0) {
    fmt::print("ERROR: no entries in tree\n");
    fh.close();
    return arrCanvases;
  }

  constexpr int nCRUs = CRU::MaxCRU;
  constexpr int nTimeBins = cmv::NTimeBinsPerTF;

  TH2F* h2d = new TH2F("hCMVvsTimeBin", ";Timebin (200 ns);Common Mode Values (ADC)",
                       100, 0, nTimeBins,
                       110, -100.5, 9.5);
  h2d->SetDirectory(nullptr);
  h2d->SetStats(1);
  TH1F* h1d = new TH1F("hCMV", ";Common Mode Values (ADC);Counts",
                       110, -100.5, 9.5);
  h1d->SetDirectory(nullptr);
  h1d->SetStats(1);

  TH1F* h1dCRU = new TH1F("hCRU", ";CRU;Counts",
                          360, -0.5, 359.5);
  h1dCRU->SetDirectory(nullptr);
  h1dCRU->SetStats(1);
  TH2F* h2dCRU = new TH2F("hCMVvsCRU", ";CRU;Common Mode Values (ADC)",
                          360, -0.5, 359.5,
                          110, -100.5, 9.5);
  h2dCRU->SetDirectory(nullptr);
  h2dCRU->SetStats(0);

  fmt::print("Branch format: {}\n", fh.isCompressed ? "CMVPerTFCompressed" : "CMVPerTF (raw)");

  long firstOrbit = -1;
  long firstOrbitDPL = -1;

  // Pre-allocate fill arrays once; x-values (timebins) are constant across entries and CRUs
  const int fillsPerEntry = nCRUs * nTimeBins;
  std::vector<double> xArr(fillsPerEntry), yArr(fillsPerEntry), wArr(fillsPerEntry, 1.0), cruArr(fillsPerEntry);
  for (int cru = 0; cru < nCRUs; ++cru) {
    for (int tb = 0; tb < nTimeBins; ++tb) {
      xArr[cru * nTimeBins + tb] = tb;
      cruArr[cru * nTimeBins + tb] = cru;
    }
  }

  for (int i = 0; i < nEntries; ++i) {
    const CMVPerTF* tf = fh.getEntry(i);
    if (!tf) {
      continue;
    }

    firstOrbit = tf->firstOrbit;
    firstOrbitDPL = tf->firstOrbitDPL;

    fmt::print("Entry {}: firstOrbit: {}, firstOrbitDPL: {}\n", i, firstOrbit, firstOrbitDPL);

    for (int cru = 0; cru < nCRUs; ++cru) {
      for (int tb = 0; tb < nTimeBins; ++tb) {
        yArr[cru * nTimeBins + tb] = tf->getCMVFloat(cru, tb);
        // fmt::print("Entry {}: cru: {}, tb: {}, cmv: {}\n", i, cru, tb, tf->getCMVFloat(cru, tb));
      }
    }
    h2d->FillN(fillsPerEntry, xArr.data(), yArr.data(), wArr.data());
    h1d->FillN(fillsPerEntry, yArr.data(), wArr.data());
    h2dCRU->FillN(fillsPerEntry, cruArr.data(), yArr.data(), wArr.data());
    h1dCRU->FillN(fillsPerEntry, cruArr.data(), wArr.data());
  }

  fh.close();

  // draw
  auto* c = new TCanvas("cCMVvsTimeBin", "");
  c->SetLogz();
  h2d->Draw("colz");

  arrCanvases->Add(c);

  auto* c1 = new TCanvas("cCMVDistribution", "");
  c1->SetLogy();
  h1d->Draw();

  arrCanvases->Add(c1);

  auto* c2 = new TCanvas("cCRUDistribution", "");
  h1dCRU->Draw();

  arrCanvases->Add(c2);

  auto* c3 = new TCanvas("cCMVvsCRU", "");
  c3->SetLogz();
  h2dCRU->Draw("colz");

  arrCanvases->Add(c3);

  if (outDir.size()) {
    utils::saveCanvases(*arrCanvases, outDir, "", rootFileName);
  }

  return arrCanvases;
}
