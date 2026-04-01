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
#include <fmt/format.h>

#include "TFile.h"
#include "TParameter.h"
#include "TTree.h"
#include "TH2F.h"
#include "TCanvas.h"

#include "TPCCalibration/CMVContainer.h"
#include "TPCBase/Utils.h"
#endif

using namespace o2::tpc;

/// Draw CMV (Common Mode Values) vs timebin from a CCDB TTree file
/// \param filename  input ROOT file containing the ccdb_object TTree
/// \param outDir    output directory for saved plots; nothing is saved if empty
/// \return          array of canvases
TObjArray* drawCMV(std::string_view filename, std::string_view outDir)
{
  TObjArray* arrCanvases = new TObjArray;
  arrCanvases->SetName("CMV");

  // open file
  TFile f(filename.data(), "READ");
  if (f.IsZombie()) {
    fmt::print("ERROR: cannot open '{}'\n", filename);
    return arrCanvases;
  }
  fmt::print("Opened file: {}\n", filename);

  // get TTree
  TTree* tree = nullptr;
  f.GetObject("ccdb_object", tree);
  if (!tree) {
    fmt::print("ERROR: TTree 'ccdb_object' not found\n");
    return arrCanvases;
  }
  fmt::print("Tree 'ccdb_object' found, entries: {}\n", tree->GetEntries());

  // read metadata
  long firstTF = -1, lastTF = -1;
  if (auto* userInfo = tree->GetUserInfo()) {
    for (int i = 0; i < userInfo->GetSize(); ++i) {
      if (auto* p = dynamic_cast<TParameter<long>*>(userInfo->At(i))) {
        if (std::string(p->GetName()) == "firstTF")
          firstTF = p->GetVal();
        if (std::string(p->GetName()) == "lastTF")
          lastTF = p->GetVal();
      }
    }
  }
  fmt::print("firstTF: {}, lastTF: {}\n", firstTF, lastTF);

  const int nEntries = tree->GetEntries();
  if (nEntries == 0) {
    fmt::print("ERROR: no entries in tree\n");
    return arrCanvases;
  }

  constexpr int nCRUs = CRU::MaxCRU;
  constexpr int nTimeBins = cmv::NTimeBinsPerTF;

  TH2F* h2d = new TH2F("hCMVvsTimeBin", ";Timebin (200 ns);Common Mode Values (ADC)",
                       100, 0, nTimeBins,
                       110, -100.5, 9.5);
  h2d->SetStats(0);

  // auto-detect branch format: sparse, huffman, compressed (delta+zigzag+varint), or raw CMVPerTF
  const bool isSparse = (tree->GetBranch("CMVPerTFSparse") != nullptr);
  const bool isHuffman = (tree->GetBranch("CMVPerTFHuffman") != nullptr);
  const bool isCompressed = (tree->GetBranch("CMVPerTFCompressed") != nullptr);
  const bool isRaw = (tree->GetBranch("CMVPerTF") != nullptr);
  if (!isSparse && !isHuffman && !isCompressed && !isRaw) {
    fmt::print("ERROR: no recognised branch found (expected CMVPerTFSparse, CMVPerTFHuffman, CMVPerTFCompressed, or CMVPerTF)\n");
    return arrCanvases;
  }
  const std::string branchFormat = isSparse ? "CMVPerTFSparse (sparse)" : (isHuffman ? "CMVPerTFHuffman (delta+zigzag+Huffman)" : (isCompressed ? "CMVPerTFCompressed (delta+zigzag+varint)" : "CMVPerTF (raw)"));
  fmt::print("Branch format: {}\n", branchFormat);

  // branch setup — only one pointer is active depending on the detected format
  o2::tpc::CMVPerTFSparse* tfSparse = nullptr;
  o2::tpc::CMVPerTFHuffman* tfHuffman = nullptr;
  o2::tpc::CMVPerTFCompressed* tfCompressed = nullptr;
  o2::tpc::CMVPerTF* tfRaw = nullptr;
  // staging object for decompression
  CMVPerTF* tfDecoded = (isSparse || isHuffman || isCompressed) ? new CMVPerTF() : nullptr;

  if (isSparse) {
    tree->SetBranchAddress("CMVPerTFSparse", &tfSparse);
  } else if (isHuffman) {
    tree->SetBranchAddress("CMVPerTFHuffman", &tfHuffman);
  } else if (isCompressed) {
    tree->SetBranchAddress("CMVPerTFCompressed", &tfCompressed);
  } else {
    tree->SetBranchAddress("CMVPerTF", &tfRaw);
  }

  long firstOrbit = -1;

  for (int i = 0; i < nEntries; ++i) {
    tree->GetEntry(i);

    // resolve to a unified pointer regardless of storage format
    const CMVPerTF* tf = nullptr;
    if (isSparse) {
      tfSparse->decompress(tfDecoded);
      tf = tfDecoded;
    } else if (isHuffman) {
      tfHuffman->decompress(tfDecoded);
      tf = tfDecoded;
    } else if (isCompressed) {
      tfCompressed->decompress(tfDecoded);
      tf = tfDecoded;
    } else {
      tf = tfRaw;
    }

    if (i == 0) {
      firstOrbit = tf->firstOrbit;
    }

    for (int cru = 0; cru < nCRUs; ++cru) {
      for (int tb = 0; tb < nTimeBins; ++tb) {
        h2d->Fill(tb, tf->getCMVFloat(cru, tb));
      }
    }
  }

  delete tfDecoded;
  tree->ResetBranchAddresses();
  delete tfSparse;
  delete tfHuffman;
  delete tfCompressed;

  fmt::print("firstOrbit: {}\n", firstOrbit);

  // draw
  auto* c = new TCanvas("cCMVvsTimeBin", "");
  c->SetLogz();
  h2d->Draw("colz");

  arrCanvases->Add(c);

  if (outDir.size()) {
    utils::saveCanvases(*arrCanvases, outDir, "png,pdf", "CMVCanvases.root");
  }

  f.Close();
  return arrCanvases;
}
