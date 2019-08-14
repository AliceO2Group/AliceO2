// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test TreeStream
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <TFile.h>
#include <TRandom.h>
#include <TVectorD.h>
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "CommonUtils/TreeStream.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CommonUtils/RootChain.h"
#include "ReconstructionDataFormats/Track.h"
#include <FairLogger.h>
#include <string>

using namespace o2::utils;

bool UnitTestSparse(Double_t scale, Int_t testEntries);

BOOST_AUTO_TEST_CASE(TreeStream_test)
{
  // Example test function to show functionality of TreeStreamRedirector

  // create the  redirector associated with file (testredirector.root)
  FairLogger* logger = FairLogger::GetLogger();

  LOG(INFO) << "Testing  TreeStream creation";
  std::string outFName("testTreeStream.root");
  int nit = 50;
  {
    TreeStreamRedirector tstStream(outFName.data(), "recreate");
    // write tree named TrackTree of int counter, float  and TrackParCov (using pointer)
    // and another similar tree but using reference of TrackParCov
    std::array<float, o2::track::kNParams> par{};
    for (int i = 0; i < nit; i++) {
      par[o2::track::kQ2Pt] = 0.5 + float(i) / nit;
      float x = 10. + float(i) / nit * 200.;
      o2::track::TrackPar trc(0., 0., par);
      trc.propagateParamTo(x, 0.5);
      tstStream << "TrackTree"
                << "id=" << i << "x=" << x << "track=" << &trc << "\n";

      tstStream << "TrackTreeR"
                << "id=" << i << "x=" << x << "track=" << trc << "\n";
    }
    // on destruction of tstTreem the trees will be stored, but we can also force it by
    tstStream.Close();
  }
  //
  LOG(INFO) << "Testing reading back tree maid by the TreeStream ";
  // read back tracks
  {
    TFile inpf(outFName.data());
    BOOST_CHECK(!inpf.IsZombie());
    auto tree = (TTree*)inpf.GetObjectChecked("TrackTree", "TTree");
    BOOST_CHECK(tree);
    int nent = tree->GetEntries();
    BOOST_CHECK(nent == nit);
    int id;
    float x;
    o2::track::TrackPar* trc = nullptr;
    BOOST_CHECK(!tree->SetBranchAddress("id", &id));
    BOOST_CHECK(!tree->SetBranchAddress("x", &x));
    BOOST_CHECK(!tree->SetBranchAddress("track", &trc));

    for (int i = 0; i < nent; i++) {
      tree->GetEntry(i);
      BOOST_CHECK(id == i);
      LOG(INFO) << "id: " << id << " X: " << x << " Track> ";
      trc->printParam();
      BOOST_CHECK(std::abs(x - trc->getX()) < 1e-4);
    }
  }

  LOG(INFO) << "Testing loading tree via RootChain";
  //
  auto chain = RootChain::load("TrackTree", outFName);
  BOOST_CHECK(chain->GetEntries());
  chain->Print();

  // we can also write the stream to external file open in write mode:
  {
    TFile inpf(outFName.data(), "update");
    TreeStream strm("TreeNamed");
    for (int i = 0; i < nit; i++) {
      TNamed nm(Form("obj%d", i), "");
      strm << "idx=" << i << "named=" << &nm << "\n";
    }
    strm.Close(); // flush the tree
  }

  // run Marian's old unit test
  LOG(INFO) << "Doing  UnitTestSparse";
  nit = 1000;
  BOOST_CHECK(UnitTestSparse(0.5, nit));
  BOOST_CHECK(UnitTestSparse(0.1, nit));
  //
}

//_________________________________________________
bool UnitTestSparse(Double_t scale, Int_t testEntries)
{
  // Unit test for the TreeStreamRedirector
  // 1.) Test TTreeRedirector
  //      a.) Fill tree with random vectors
  //      b.) Fill downscaled version of vectors
  //      c.) The same skipping first entry
  // 2.) Check results wtitten to terminale
  //     a.) Disk consumption
  //             skip data should be scale time smaller than full
  //             zerro replaced  ata should be compresed time smaller than full
  //     b.) Test invariants
  // Input parameter scale => downscaling of sprse element

  std::string outFName("testTreeStreamSparse.root");
  if (scale <= 0)
    scale = 1;
  if (scale > 1)
    scale = 1;
  TreeStreamRedirector* pcstream = new TreeStreamRedirector(outFName.data(), "recreate");
  for (Int_t ientry = 0; ientry < testEntries; ientry++) {
    TVectorD vecRandom(200);
    TVectorD vecZerro(200); // zerro vector
    for (Int_t j = 0; j < 200; j++)
      vecRandom[j] = j + ientry + 0.1 * gRandom->Rndm();
    Bool_t isSelected = (gRandom->Rndm() < scale);
    TVectorD* pvecFull = &vecRandom;
    TVectorD* pvecSparse = isSelected ? &vecRandom : nullptr;
    TVectorD* pvecSparse0 = isSelected ? &vecRandom : nullptr;
    TVectorD* pvecSparse1 = isSelected ? &vecRandom : &vecZerro;

    if (ientry == 0) {
      pvecSparse0 = nullptr;
      pvecSparse = &vecRandom;
    }
    (*pcstream) << "Full" << // stored all vectors
      "ientry=" << ientry << "vec.=" << pvecFull << "\n";
    (*pcstream) << "SparseSkip" << // fraction of vectors stored
      "ientry=" << ientry << "vec.=" << pvecSparse << "\n";
    (*pcstream) << "SparseSkip0" << // fraction with -pointer
      "ientry=" << ientry << "vec.=" << pvecSparse0 << "\n";
    (*pcstream) << "SparseZerro" << // all vectors filled, franction filled with 0
      "ientry=" << ientry << "vec.=" << pvecSparse1 << "\n";
  }
  delete pcstream;
  //
  // 2.) check results
  //

  TFile* f = TFile::Open(outFName.data());
  if (!f) {
    printf("Failed to open file: %s\n", outFName.data());
    return false;
  }
  TTree* treeFull = (TTree*)f->Get("Full");
  TTree* treeSparseSkip = (TTree*)f->Get("SparseSkip");
  TTree* treeSparseSkip0 = (TTree*)f->Get("SparseSkip0");
  TTree* treeSparseZerro = (TTree*)f->Get("SparseZerro");
  //    a.) data volume
  //
  Double_t ratio = (1. / scale) * treeSparseSkip->GetZipBytes() / Double_t(treeFull->GetZipBytes());
  Double_t ratio0 = (1. / scale) * treeSparseSkip0->GetZipBytes() / Double_t(treeFull->GetZipBytes());
  Double_t ratio1 = (1. / scale) * treeSparseZerro->GetZipBytes() / Double_t(treeFull->GetZipBytes());
  printf("#UnitTest:\tTestSparse(%f)\tRatioSkip\t%f\n", scale, ratio);
  printf("#UnitTest:\tTestSparse(%f)\tRatioSkip0\t%f\n", scale, ratio0);
  printf("#UnitTest:\tTestSparse(%f)\tRatioZerro\t%f\n", scale, ratio1);
  //    b.) Integrity
  Int_t outlyersSparseSkip = treeSparseSkip->Draw("1", "(vec.fElements-ientry-Iteration$-0.5)>0.5", "goff");
  Int_t outlyersSparseSkip0 = treeSparseSkip0->Draw("1", "(vec.fElements-ientry-Iteration$-0.5)>0.5", "goff");
  printf("#UnitTest:\tTestSparse(%f)\tOutlyersSkip\t%d\n", scale, outlyersSparseSkip != 0);
  printf("#UnitTest:\tTestSparse(%f)\tOutlyersSkip0\t%d\n", scale, outlyersSparseSkip0 != 0);
  //    c.) Number of entries
  //
  Int_t entries = treeFull->GetEntries();
  Int_t entries0 = treeSparseSkip0->GetEntries();
  Bool_t isOKStat = (entries == entries0);
  printf("#UnitTest:\tTestSparse(%f)\tEntries\t%d\n", scale, isOKStat);
  //
  //   d.)Reading test
  TVectorD* pvecRead = nullptr;
  treeSparseSkip0->SetBranchAddress("vec.", &pvecRead);
  Bool_t readOK = kTRUE;
  for (Int_t ientry = 0; ientry < testEntries; ientry++) {
    if (!pvecRead)
      continue;
    if (pvecRead->GetNrows() == 0)
      continue;
    if (TMath::Abs((*pvecRead)[0] - ientry) > 0.5)
      readOK = kFALSE;
  }
  printf("#UnitTest:\tTestSparse(%f)\tReadOK\t%d\n", scale, readOK);
  //
  //   e.)Global test
  Bool_t isOK = (outlyersSparseSkip0 == 0) && isOKStat && readOK;
  printf("#UnitTest:\tTestSparse(%f)\tisOk\t%d\n", scale, isOK);

  return isOK;
}
