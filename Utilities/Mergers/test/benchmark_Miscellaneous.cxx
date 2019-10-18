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
#include <THn.h>
#include <TTree.h>
#include <THnSparse.h>
#include <TF1.h>
#include <TF2.h>
#include <TF3.h>
#include <TRandom.h>
#include <TRandomGen.h>

#include <boost/histogram.hpp>

static void BM_dynamic_cast(benchmark::State& state)
{
  TCollection* collection = new TList();
  collection->SetOwner(true);
  TH1I* h = new TH1I("histo", "histo", 1000, 0, 1000000);
  collection->Add(h);

  TObject* collectionObj = collection;
  size_t entries = 0;

  for (auto _ : state) {
    if (auto* list = dynamic_cast<TList*>(collectionObj)) {
      entries += list->GetEntries();
    }
  }

  (void)entries;
  delete collection;
}

static void BM_InheritsFrom(benchmark::State& state)
{
  TCollection* collection = new TList();
  collection->SetOwner(true);
  TH1I* h = new TH1I("histo", "histo", 1000, 0, 1000000);
  collection->Add(h);

  TObject* collectionObj = collection;
  size_t entries = 0;
  for (auto _ : state) {
    if (collectionObj->InheritsFrom("TCollection")) {
      entries += reinterpret_cast<TList*>(collectionObj)->GetEntries();
    }
  }

  (void)entries;
  delete collection;
}

static void BM_MergeCollectionOnHeap(benchmark::State& state)
{
  TCollection* collection = new TObjArray();
  collection->SetOwner(true);
  TF1* uni = new TF1("uni", "1", 0, 1000000);
  TH1I* h = new TH1I("test", "test", 100, 0, 1000000);
  TH1I* m = new TH1I("merged", "merged", 100, 0, 1000000);
  h->FillRandom("uni", 50000);
  collection->Add(h);

  for (auto _ : state) {
    m->Merge(collection);
  }

  delete collection;
  delete uni;
  delete m;
}

static void BM_MergeCollectionOnStack(benchmark::State& state)
{
  TObjArray collection;
  collection.SetOwner(true);
  TF1* uni = new TF1("uni", "1", 0, 1000000);
  TH1I* h = new TH1I("test", "test", 100, 0, 1000000);
  TH1I* m = new TH1I("merged", "merged", 100, 0, 1000000);
  h->FillRandom("uni", 50000);
  collection.Add(h);

  for (auto _ : state) {
    m->Merge(&collection);
  }

  delete uni;
  delete m;
}

static void BM_MergeWithoutNoCheck(benchmark::State& state)
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
    m->Merge(collection);
  }

  delete collection;
  delete m;
  delete uni;
}

static void BM_MergeWithNoCheck(benchmark::State& state)
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
    m->Merge(collection);
  }

  delete collection;
  delete m;
  delete uni;
}

// TEST: See which way of casting ROOT objects is faster.
// RESULT: dynamic_cast wins by an order of magnitude.
BENCHMARK(BM_dynamic_cast);
BENCHMARK(BM_InheritsFrom);
// TEST: See if there is a difference between merging TH1 (on heap) stored in a collection on stack vs on heap.
// RESULT: The difference is negligible.
BENCHMARK(BM_MergeCollectionOnHeap);
BENCHMARK(BM_MergeCollectionOnStack);
// TEST: see if -NOCHECK indeed improves the performance
// RESULT: Probably barely
BENCHMARK(BM_MergeWithoutNoCheck)->Arg(1)->Arg(100);
BENCHMARK(BM_MergeWithNoCheck)->Arg(1)->Arg(100);

BENCHMARK_MAIN();
