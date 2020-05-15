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

#include <boost/histogram.hpp>

namespace bh = boost::histogram;

#include <ctime>

static void BM_mergingCollectionsTH1I(benchmark::State& state)
{
  size_t collectionSize = state.range(0);
  size_t bins = 62500; // makes 250kB

  TCollection* collection = new TObjArray();
  collection->SetOwner(true);
  TF1* uni = new TF1("uni", "1", 0, 1000000);
  for (size_t i = 0; i < collectionSize; i++) {
    TH1I* h = new TH1I(("test" + std::to_string(i)).c_str(), "test", bins, 0, 1000000);
    h->FillRandom("uni", 50000);
    collection->Add(h);
  }

  TH1I* m = new TH1I("merged", "merged", bins, 0, 1000000);

  for (auto _ : state) {
    m->Merge(collection, "-NOCHECK");
  }

  delete collection;
  delete m;
  delete uni;
}

static void BM_mergingCollectionsTH2I(benchmark::State& state)
{

  size_t collectionSize = state.range(0);
  size_t bins = 250; // 250 bins * 250 bins * 4B makes 250kB

  TCollection* collection = new TObjArray();
  collection->SetOwner(true);
  TF2* uni = new TF2("uni", "1", 0, 1000000, 0, 1000000);
  for (size_t i = 0; i < collectionSize; i++) {
    TH2I* h = new TH2I(("test" + std::to_string(i)).c_str(), "test", bins, 0, 1000000, bins, 0, 1000000);
    h->FillRandom("uni", 50000);
    collection->Add(h);
  }

  TH2I* m = new TH2I("merged", "merged", bins, 0, 1000000, bins, 0, 1000000);

  for (auto _ : state) {
    m->Merge(collection, "-NOCHECK");
  }

  delete collection;
  delete m;
}

static void BM_mergingCollectionsTH3I(benchmark::State& state)
{
  size_t collectionSize = state.range(0);
  size_t bins = 40; // 40 bins * 40 bins * 40 bins * 4B makes 256kB

  TCollection* collection = new TObjArray();
  collection->SetOwner(true);
  TF3* uni = new TF3("uni", "1", 0, 1000000, 0, 1000000, 0, 1000000);
  for (size_t i = 0; i < collectionSize; i++) {
    TH3I* h = new TH3I(("test" + std::to_string(i)).c_str(), "test",
                       bins, 0, 1000000,
                       bins, 0, 1000000,
                       bins, 0, 1000000);
    h->FillRandom("uni", 50000);
    collection->Add(h);
  }

  TH3I* m = new TH3I("merged", "merged", bins, 0, 1000000, bins, 0, 1000000, bins, 0, 1000000);

  for (auto _ : state) {
    m->Merge(collection, "-NOCHECK");
  }

  delete collection;
  delete m;
}

static void BM_mergingCollectionsTHNSparse(benchmark::State& state)
{
  size_t collectionSize = state.range(0);

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

  for (auto _ : state) {

    state.PauseTiming();
    TCollection* collection = new TObjArray();
    collection->SetOwner(true);
    for (size_t i = 0; i < collectionSize; i++) {

      auto* h = new THnSparseI(("test" + std::to_string(i)).c_str(), "test", dim, binsDims, mins, maxs);
      for (size_t entry = 0; entry < 50000; entry++) {
        gen.RndmArray(dim, randomArray);
        for (double& r : randomArray) {
          r *= max;
        }
        h->Fill(randomArray);
      }
      collection->Add(h);
    }
    auto* m = new THnSparseI("merged", "merged", dim, binsDims, mins, maxs);

    state.ResumeTiming();
    m->Merge(collection);

    state.PauseTiming();

    delete collection;
    delete m;
  }
}

static void BM_mergingCollectionsTTree(benchmark::State& state)
{
  size_t collectionSize = state.range(0);

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

  for (auto _ : state) {
    state.PauseTiming();
    TCollection* collection = new TObjArray();
    collection->SetOwner(true);
    for (size_t i = 0; i < collectionSize; i++) {
      TTree* t = createTree(std::to_string(i));
      for (size_t entry = 0; entry < 6250; entry++) {
        gen.RndmArray(5, randomArray);
        branch1 = {static_cast<Int_t>(randomArray[0]), static_cast<Long64_t>(randomArray[1]), static_cast<Float_t>(randomArray[2]), randomArray[3]};
        branch2 = randomArray[4];
        t->Fill();
      }
      collection->Add(t);
    }
    TTree* merged = createTree("merged");

    state.ResumeTiming();
    merged->Merge(collection);

    state.PauseTiming();

    delete collection;
    delete merged;
  }
}

static void BM_mergingPODCollections(benchmark::State& state)
{
  size_t collectionSize = state.range(0);
  size_t bins = 62500; // makes 250kB

  std::vector<std::vector<float>*> collection;
  TF1* uni = new TF1("uni", "1", 0, 1000000);

  const size_t randoms = 50000;
  TRandomMixMax gen;
  gen.SetSeed(std::random_device()());
  Double_t randomArray[randoms];

  for (size_t i = 0; i < collectionSize; i++) {
    auto* v = new std::vector<float>(bins, 0);
    gen.RndmArray(randoms, randomArray);
    for (double r : randomArray) {
      size_t idx = r * bins;
      if (idx != bins) {
        (*v)[idx] += 1;
      }
    }
    collection.push_back(v);
  }

  auto* m = new std::vector<float>(bins, 0);

  auto merge = [&](size_t i) {
    auto* v = collection[i];
    for (size_t b = 0; b < bins; b++) {
      (*m)[b] += (*v)[b];
    }
  };

  for (auto _ : state) {
    for (size_t i = 0; i < collectionSize; i++) {
      merge(i);
    }
  }

  for (size_t i = 0; i < collectionSize; i++) {
    delete collection[i];
  }
  delete m;
  delete uni;
}

static void BM_mergingBoostRegular1DCollections(benchmark::State& state)
{
  const double min = 0.0;
  const double max = 1000000.0;
  const size_t collectionSize = state.range(0);
  const size_t bins = 62500; // makes 250kB

  auto merged = bh::make_histogram(bh::axis::regular<>(bins, min, max, "x"));

  std::vector<decltype(merged)> collection;
  TF1* uni = new TF1("uni", "1", 0, 1000000);

  const size_t randoms = 50000;
  TRandomMixMax gen;
  gen.SetSeed(std::random_device()());
  Double_t randomArray[randoms];

  for (size_t i = 0; i < collectionSize; i++) {
    collection.emplace_back(std::move(bh::make_histogram(bh::axis::regular<>(bins, min, max, "x"))));

    auto& h = collection.back();
    static_assert(std::is_reference<decltype(h)>::value);

    gen.RndmArray(randoms, randomArray);
    for (double r : randomArray) {
      h(r * max);
    }
  }

  auto merge = [&](size_t i) {
    merged += collection[i];
  };

  for (auto _ : state) {
    for (size_t i = 0; i < collectionSize; i++) {
      merge(i);
    }
  }

  delete uni;
}

static void BM_mergingBoostRegular2DCollections(benchmark::State& state)
{
  const double min = 0.0;
  const double max = 1000000.0;
  const size_t collectionSize = state.range(0);
  const size_t bins = 250; // 250 bins * 250 bins * 4B makes 250kB

  auto merged = bh::make_histogram(bh::axis::regular<>(bins, min, max, "x"), bh::axis::regular<>(bins, min, max, "y"));

  std::vector<decltype(merged)> collection;
  TF1* uni = new TF1("uni", "1", 0, 1000000);

  const size_t randoms = 50000;
  TRandomMixMax gen;
  gen.SetSeed(std::random_device()());
  Double_t randomArrayX[randoms];
  Double_t randomArrayY[randoms];

  for (size_t i = 0; i < collectionSize; i++) {
    collection.emplace_back(std::move(bh::make_histogram(bh::axis::regular<>(bins, min, max, "x"), bh::axis::regular<>(bins, min, max, "y"))));

    auto& h = collection.back();
    static_assert(std::is_reference<decltype(h)>::value);

    gen.RndmArray(randoms, randomArrayX);
    gen.RndmArray(randoms, randomArrayY);
    for (size_t r = 0; r < randoms; r++) {
      h(randomArrayX[r] * max, randomArrayY[r] * max);
    }
  }

  auto merge = [&](size_t i) {
    merged += collection[i];
  };

  for (auto _ : state) {
    for (size_t i = 0; i < collectionSize; i++) {
      merge(i);
    }
  }

  delete uni;
}

// one by one comparison
BENCHMARK(BM_mergingCollectionsTH1I)->Arg(1);
BENCHMARK(BM_mergingCollectionsTH2I)->Arg(1);
BENCHMARK(BM_mergingCollectionsTH3I)->Arg(1);
BENCHMARK(BM_mergingCollectionsTHNSparse)->Arg(1);
BENCHMARK(BM_mergingPODCollections)->Arg(1);
BENCHMARK(BM_mergingBoostRegular1DCollections)->Arg(1);
BENCHMARK(BM_mergingBoostRegular2DCollections)->Arg(1);
BENCHMARK(BM_mergingCollectionsTTree)->Arg(1);

// collections
#define BENCHMARK_RANGE_COLLECTIONS Arg(1)->Arg(1 << 2)->Arg(1 << 4)->Arg(1 << 6)->Arg(1 << 8)->Arg(1 << 10)

BENCHMARK(BM_mergingCollectionsTH1I)->BENCHMARK_RANGE_COLLECTIONS;
BENCHMARK(BM_mergingCollectionsTH2I)->BENCHMARK_RANGE_COLLECTIONS;
BENCHMARK(BM_mergingCollectionsTH3I)->BENCHMARK_RANGE_COLLECTIONS;
BENCHMARK(BM_mergingCollectionsTHNSparse)->BENCHMARK_RANGE_COLLECTIONS;
BENCHMARK(BM_mergingPODCollections)->BENCHMARK_RANGE_COLLECTIONS;
BENCHMARK(BM_mergingBoostRegular1DCollections)->BENCHMARK_RANGE_COLLECTIONS;
BENCHMARK(BM_mergingBoostRegular2DCollections)->BENCHMARK_RANGE_COLLECTIONS;
BENCHMARK(BM_mergingCollectionsTTree)->BENCHMARK_RANGE_COLLECTIONS;

BENCHMARK_MAIN();
