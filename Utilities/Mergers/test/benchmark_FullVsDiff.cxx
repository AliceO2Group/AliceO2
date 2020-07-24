// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <benchmark/benchmark.h>

#include <TObjArray.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TTree.h>
#include <THnSparse.h>
#include <TF1.h>
#include <TF2.h>
#include <TF3.h>
#include <TRandom.h>
#include <TRandomGen.h>
#include <chrono>
#include <ctime>

const size_t entriesInDiff = 50;
const size_t entriesInFull = 5000;
const size_t collectionSize = 100;

#define DIFF_OBJECTS 0
#define FULL_OBJECTS 1

static void BM_MergingTH1I(benchmark::State& state)
{
  const size_t entries = state.range(0) == FULL_OBJECTS ? entriesInFull : entriesInDiff;
  const size_t bins = 62500; // makes 250kB

  TCollection* collection = new TObjArray();
  collection->SetOwner(true);
  TF1* uni = new TF1("uni", "1", 0, 1000000);
  for (size_t i = 0; i < collectionSize; i++) {
    TH1I* h = new TH1I(("test" + std::to_string(i)).c_str(), "test", bins, 0, 1000000);
    h->FillRandom("uni", entries);
    collection->Add(h);
  }

  TH1I* m = new TH1I("merged", "merged", bins, 0, 1000000);
  // avoid memory overcommitment by doing something with data.
  for (size_t i = 0; i < bins; i++) {
    m->SetBinContent(i, 1);
  }

  for (auto _ : state) {
    if (state.range(0) == FULL_OBJECTS) {
      m->Reset();
    }

    auto start = std::chrono::high_resolution_clock::now();
    m->Merge(collection, "-NOCHECK");
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  delete collection;
  delete m;
  delete uni;
}

static void BM_MergingTH2I(benchmark::State& state)
{
  const size_t entries = state.range(0) == FULL_OBJECTS ? entriesInFull : entriesInDiff;
  size_t bins = 250; // 250 bins * 250 bins * 4B makes 250kB

  TCollection* collection = new TObjArray();
  collection->SetOwner(true);
  TF2* uni = new TF2("uni", "1", 0, 1000000, 0, 1000000);
  for (size_t i = 0; i < collectionSize; i++) {
    TH2I* h = new TH2I(("test" + std::to_string(i)).c_str(), "test", bins, 0, 1000000, bins, 0, 1000000);
    h->FillRandom("uni", entries);
    collection->Add(h);
  }

  TH2I* m = new TH2I("merged", "merged", bins, 0, 1000000, bins, 0, 1000000);
  // avoid memory overcommitment by doing something with data.
  for (size_t i = 0; i < bins; i++) {
    m->SetBinContent(i, 1);
  }

  for (auto _ : state) {
    if (state.range(0) == FULL_OBJECTS) {
      m->Reset();
    }

    auto start = std::chrono::high_resolution_clock::now();
    m->Merge(collection, "-NOCHECK");
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  delete collection;
  delete m;
}

static void BM_MergingTH3I(benchmark::State& state)
{
  const size_t entries = state.range(0) == FULL_OBJECTS ? entriesInFull : entriesInDiff;
  size_t bins = 40; // 40 bins * 40 bins * 40 bins * 4B makes 256kB

  TCollection* collection = new TObjArray();
  collection->SetOwner(true);
  TF3* uni = new TF3("uni", "1", 0, 1000000, 0, 1000000, 0, 1000000);
  for (size_t i = 0; i < collectionSize; i++) {
    TH3I* h = new TH3I(("test" + std::to_string(i)).c_str(), "test",
                       bins, 0, 1000000,
                       bins, 0, 1000000,
                       bins, 0, 1000000);
    h->FillRandom("uni", entries);
    collection->Add(h);
  }

  TH3I* m = new TH3I("merged", "merged", bins, 0, 1000000, bins, 0, 1000000, bins, 0, 1000000);
  // avoid memory overcommitment by doing something with data.
  for (size_t i = 0; i < bins; i++) {
    m->SetBinContent(i, 1);
  }

  for (auto _ : state) {
    if (state.range(0) == FULL_OBJECTS) {
      m->Reset();
    }

    auto start = std::chrono::high_resolution_clock::now();
    m->Merge(collection, "-NOCHECK");
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  delete collection;
  delete m;
}

static void BM_MergingTHnSparse(benchmark::State& state)
{
  const size_t entries = state.range(0) == FULL_OBJECTS ? entriesInFull : entriesInDiff;

  const Double_t min = 0.0;
  const Double_t max = 1000000.0;
  const size_t dim = 10;
  const Int_t bins = 250;
  const Int_t binsDims[dim] = {bins, bins, bins, bins, bins, bins, bins, bins, bins, bins};
  const Double_t mins[dim] = {min, min, min, min, min, min, min, min, min, min};
  const Double_t maxs[dim] = {max, max, max, max, max, max, max, max, max, max};

  TRandomMixMax gen;
  gen.SetSeed(std::random_device()());
  Double_t randomArray[dim];
  auto* m = new THnSparseI("merged", "merged", dim, binsDims, mins, maxs);
  for (auto _ : state) {

    TCollection* collection = new TObjArray();
    collection->SetOwner(true);
    for (size_t i = 0; i < collectionSize; i++) {

      auto* h = new THnSparseI(("test" + std::to_string(i)).c_str(), "test", dim, binsDims, mins, maxs);
      for (size_t entry = 0; entry < entries; entry++) {
        gen.RndmArray(dim, randomArray);
        for (double& r : randomArray) {
          r *= max;
        }
        h->Fill(randomArray);
      }
      collection->Add(h);
    }

    if (state.range(0) == FULL_OBJECTS) {
      m->Reset();
    }

    auto start = std::chrono::high_resolution_clock::now();
    m->Merge(collection);
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());

    delete collection;
  }
  delete m;
}

static void BM_MergingTTree(benchmark::State& state)
{
  const size_t entries = state.range(0) == FULL_OBJECTS ? entriesInFull : entriesInDiff;

  struct format1 {
    Int_t a;
    Long64_t b;
    Float_t c;
    Double_t d;
  } branch1;
  ULong64_t branch2;

  auto createTree = [&](std::string name) -> TTree* {
    TTree* tree = new TTree();
    tree->SetName(name.c_str());
    tree->Branch("b1", &branch1, "a/I:b/L:c/F:d/D");
    tree->Branch("b2", &branch2);
    return tree;
  };

  TRandomMixMax gen;
  gen.SetSeed(std::random_device()());
  Double_t randomArray[5];
  TTree* merged = createTree("merged");

  for (auto _ : state) {
    TCollection* collection = new TObjArray();
    collection->SetOwner(true);
    for (size_t i = 0; i < collectionSize; i++) {
      TTree* t = createTree(std::to_string(i));
      for (size_t entry = 0; entry < entries; entry++) {
        gen.RndmArray(5, randomArray);
        branch1 = {static_cast<Int_t>(randomArray[0]), static_cast<Long64_t>(randomArray[1]), static_cast<Float_t>(randomArray[2]), randomArray[3]};
        branch2 = randomArray[4];
        t->Fill();
      }
      collection->Add(t);
    }

    if (state.range(0) == FULL_OBJECTS) {
      merged->Reset();
    }

    auto start = std::chrono::high_resolution_clock::now();
    merged->Merge(collection, "-NOCHECK");
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
  delete merged;
}

BENCHMARK(BM_MergingTH1I)->Arg(DIFF_OBJECTS)->UseManualTime();
BENCHMARK(BM_MergingTH1I)->Arg(FULL_OBJECTS)->UseManualTime();
BENCHMARK(BM_MergingTH2I)->Arg(DIFF_OBJECTS)->UseManualTime();
BENCHMARK(BM_MergingTH2I)->Arg(FULL_OBJECTS)->UseManualTime();
BENCHMARK(BM_MergingTH3I)->Arg(DIFF_OBJECTS)->UseManualTime();
BENCHMARK(BM_MergingTH3I)->Arg(FULL_OBJECTS)->UseManualTime();
BENCHMARK(BM_MergingTHnSparse)->Arg(DIFF_OBJECTS)->UseManualTime();
BENCHMARK(BM_MergingTHnSparse)->Arg(FULL_OBJECTS)->UseManualTime();
BENCHMARK(BM_MergingTTree)->Arg(DIFF_OBJECTS)->UseManualTime();
BENCHMARK(BM_MergingTTree)->Arg(FULL_OBJECTS)->UseManualTime();

BENCHMARK_MAIN();