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
#include <TMessage.h>

#include "Framework/TMessageSerializer.h"

#include <type_traits>

#include <iostream>
#include <fstream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/histogram.hpp>
#include <boost/histogram/serialization.hpp>
#include <sstream>

namespace bh = boost::histogram;

enum class Measurement {
  Size,
  SizeAfterSerialisation,
  Deserialisation,
  Merging,
  Serialisation
};

struct Results {
  size_t sizeBytes = 0;
  size_t sizeSerialisedBytes = 0;
  double deserialisationSeconds = 0;
  double mergingSeconds = 0;
  double serialisationSeconds = 0;
};

struct Parameters {
  constexpr static Parameters forHistograms(size_t objectSize, size_t entries)
  {
    return {objectSize, 0, 0, entries};
  }
  constexpr static Parameters forSparse(size_t bins, size_t dimensions, size_t entries)
  {
    return {0, bins, dimensions, entries};
  }
  constexpr static Parameters forTrees(size_t branches, size_t branchSize, size_t entries)
  {
    return {0, 0, 0, entries, branches, branchSize};
  }
  size_t objectSize = 0;
  size_t bins = 0;
  size_t dimensions = 0;
  size_t entries = 0;
  size_t branches = 0;
  size_t branchSize = 0;
};

auto measure = [](Measurement m, auto* o, auto* i) -> double {
  switch (m) {
    case Measurement::Size: {
      const double scale = 1.0; //000000000.0;
      if constexpr (std::is_base_of<TObject, typename std::remove_pointer<decltype(o)>::type>::value) {
        if (o->InheritsFrom(TH1::Class())) {
          // this includes TH1, TH2, TH3
          // i don't see an easy way to find out the size of a cell, i assume that they are Int
          return reinterpret_cast<TH1*>(o)->GetNcells() * sizeof(Int_t) / scale;
        } else if (o->InheritsFrom(THnSparse::Class())) {
          auto* sparse = reinterpret_cast<THnSparse*>(o);
          // this is has to be multiplied with bin (entry) size, but we cannot get it from th einterface.
          return sparse->GetNChunks() * sparse->GetChunkSize() / scale;
        } else if (o->InheritsFrom(THnBase::Class())) {
          // this includes THn and THnSparse
          return reinterpret_cast<THnBase*>(o)->GetNbins() * sizeof(Int_t) / scale;
        } else if (o->InheritsFrom(TTree::Class())) {
          size_t totalSize = 0;
          auto tree = reinterpret_cast<TTree*>(o);
          auto branchList = tree->GetListOfBranches();
          for (const auto* branch : *branchList) {
            totalSize += dynamic_cast<const TBranch*>(branch)->GetTotalSize();
          }
          return totalSize / scale;
        } else {
          throw std::runtime_error("Object with type '" + std::string(o->ClassName()) + "' is not one of the mergeable types.");
        }
      } else {
        // boost
        return o->size() * sizeof(int);
      }
    }
    case Measurement::Merging: {
      auto end = std::chrono::high_resolution_clock::now();
      auto start = std::chrono::high_resolution_clock::now();
      if constexpr (std::is_base_of<TObject, typename std::remove_pointer<decltype(o)>::type>::value) {
        if (o->InheritsFrom(TH1::Class())) {
          // this includes TH1, TH2, TH3
          start = std::chrono::high_resolution_clock::now();
          reinterpret_cast<TH1*>(o)->Merge(i, "-NOCHECK");
        } else if (o->InheritsFrom(THnBase::Class())) {
          // this includes THn and THnSparse
          start = std::chrono::high_resolution_clock::now();
          reinterpret_cast<THnBase*>(o)->Merge(i);
        } else if (o->InheritsFrom(TTree::Class())) {
          start = std::chrono::high_resolution_clock::now();
          reinterpret_cast<TTree*>(o)->Merge(i);
        } else {
          throw std::runtime_error("Object with type '" + std::string(o->ClassName()) + "' is not one of the mergeable types.");
        }
        end = std::chrono::high_resolution_clock::now();
      } else {
        // boost
        *o += *i;
        end = std::chrono::high_resolution_clock::now();
        (void)*o;
      }
      auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
      return elapsed_seconds.count();
    }
    case Measurement::Serialisation: {
      auto end = std::chrono::high_resolution_clock::now();
      auto start = std::chrono::high_resolution_clock::now();
      if constexpr (std::is_base_of<TObject, typename std::remove_pointer<decltype(o)>::type>::value) {
        TMessage* tm = new TMessage(kMESS_OBJECT);
        tm->WriteObject(o);
        end = std::chrono::high_resolution_clock::now();
        (void)*tm;
        delete tm;
      } else {
        // boost
        std::ostringstream os;
        std::string buf;
        boost::archive::binary_oarchive oa(os);
        oa << *o;
        end = std::chrono::high_resolution_clock::now();
        (void)os; // hopefully this will prevent from optimising this code out.
      }
      auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
      return elapsed_seconds.count();
    }
    case Measurement::Deserialisation: {
      auto start = std::chrono::high_resolution_clock::now();
      auto end = std::chrono::high_resolution_clock::now();
      if constexpr (std::is_base_of<TObject, typename std::remove_pointer<decltype(o)>::type>::value) {
        TMessage* tm = new TMessage(kMESS_OBJECT);
        tm->WriteObject(o);
        start = std::chrono::high_resolution_clock::now();

        // Needed to take into account that  FairInputTBuffer expects the first 8 bytes to be the
        // allocator pointer, which is not present in the TMessage buffer.
        o2::framework::FairInputTBuffer ftm(const_cast<char*>(tm->Buffer() - 8), tm->BufferSize() + 8);
        ftm.InitMap();
        auto* storedClass = ftm.ReadClass();
        if (storedClass == nullptr) {
          throw std::runtime_error("Unknown stored class");
        }
        ftm.SetBufferOffset(0);
        ftm.ResetMap();
        auto* tObjectClass = TClass::GetClass(typeid(TObject));

        if (!storedClass->InheritsFrom(tObjectClass)) {
          throw std::runtime_error("Class '" + std::string(storedClass->GetName()) + "'does not inherit from TObject");
        }

        auto* object = ftm.ReadObjectAny(storedClass);
        if (object == nullptr) {
          throw std::runtime_error("Failed to read object with name '" + std::string(storedClass->GetName()) + "' from message using ROOT serialization.");
        }

        auto tobject = static_cast<TObject*>(object);
        end = std::chrono::high_resolution_clock::now();
        (void)*tobject;
        delete tm;
        delete tobject;
      } else {
        std::ostringstream os;
        std::string buf;
        boost::archive::binary_oarchive oa(os);
        oa << *o;
        buf = os.str();

        start = std::chrono::high_resolution_clock::now();
        auto deserialisedHistogram = typename std::remove_pointer<decltype(o)>::type();
        std::istringstream is(buf);
        boost::archive::binary_iarchive ia(is);
        ia >> deserialisedHistogram;
        end = std::chrono::high_resolution_clock::now();
        assert(deserialisedHistogram == *o);
      }
      auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
      return elapsed_seconds.count();
    }
    case Measurement::SizeAfterSerialisation: {
      const double scale = 1.0; //1000000000.0;
      if constexpr (std::is_base_of<TObject, typename std::remove_pointer<decltype(o)>::type>::value) {
        TMessage* tm = new TMessage(kMESS_OBJECT);
        tm->WriteObject(o);
        auto serialisedSize = tm->BufferSize();
        delete tm;
        return serialisedSize / scale;
      } else {
        std::ostringstream os;
        std::string buf;
        boost::archive::binary_oarchive oa(os);
        oa << *i;
        buf = os.str();
        return buf.size() / scale;
      }
    }
  }
  throw;
};

static std::vector<Results> BM_TH1I(size_t repetitions, const Parameters p)
{
  const size_t objSize = p.objectSize;
  const size_t entries = p.entries;
  const size_t bins = objSize / sizeof(Int_t);

  auto m = std::make_unique<TH1I>("merged", "merged", bins, 0, 1000000);
  // avoid memory overcommitment by doing something with data.
  for (size_t i = 0; i < bins; i++) {
    m->SetBinContent(i, 1);
  }

  std::unique_ptr<TCollection> collection = std::make_unique<TObjArray>();
  collection->SetOwner(true);
  auto uni = std::make_unique<TF1>("uni", "1", 0, 1000000);
  TH1I* h = new TH1I("test", "test", bins, 0, 1000000);
  collection->Add(h);

  std::vector<Results> allResults;
  for (size_t r = 0; r < repetitions; r++) {
    h->Reset();
    h->FillRandom("uni", entries);
    Results iterationResults;
    iterationResults.sizeBytes = measure(Measurement::Size, m.get(), collection.get());
    iterationResults.sizeSerialisedBytes = measure(Measurement::SizeAfterSerialisation, m.get(), collection.get());
    iterationResults.deserialisationSeconds = measure(Measurement::Deserialisation, m.get(), collection.get());
    iterationResults.serialisationSeconds = measure(Measurement::Serialisation, m.get(), collection.get());
    iterationResults.mergingSeconds = measure(Measurement::Merging, m.get(), collection.get());
    allResults.push_back(iterationResults);
  }
  return allResults;
}

static std::vector<Results> BM_TH2I(size_t repetitions, const Parameters p)
{
  const size_t objSize = p.objectSize;
  const size_t entries = p.entries;
  const size_t bins = std::sqrt(objSize / sizeof(Int_t));

  auto m = std::make_unique<TH2I>("merged", "merged", bins, 0, 1000000, bins, 0, 1000000);
  // avoid memory overcommitment by doing something with data.
  for (size_t i = 0; i < bins * bins; i++) {
    m->SetBinContent(i, 1);
  }

  std::unique_ptr<TCollection> collection = std::make_unique<TObjArray>();
  collection->SetOwner(true);
  auto uni = std::make_unique<TF2>("uni", "1", 0, 1000000, 0, 1000000);
  auto* h = new TH2I("test", "test", bins, 0, 1000000, bins, 0, 1000000);
  collection->Add(h);

  std::vector<Results> allResults;
  for (size_t r = 0; r < repetitions; r++) {
    h->Reset();
    h->FillRandom("uni", entries);
    Results iterationResults;
    iterationResults.sizeBytes = measure(Measurement::Size, m.get(), collection.get());
    iterationResults.sizeSerialisedBytes = measure(Measurement::SizeAfterSerialisation, m.get(), collection.get());
    iterationResults.deserialisationSeconds = measure(Measurement::Deserialisation, m.get(), collection.get());
    iterationResults.serialisationSeconds = measure(Measurement::Serialisation, m.get(), collection.get());
    iterationResults.mergingSeconds = measure(Measurement::Merging, m.get(), collection.get());
    allResults.push_back(iterationResults);
  }
  return allResults;
}

static std::vector<Results> BM_TH3I(size_t repetitions, const Parameters p)
{
  const size_t objSize = p.objectSize;
  const size_t entries = p.entries;
  const size_t bins = std::pow(objSize / sizeof(Int_t), 1 / 3.0);

  auto m = std::make_unique<TH3I>("merged", "merged", bins, 0, 1000000, bins, 0, 1000000, bins, 0, 1000000);
  // avoid memory overcommitment by doing something with data.
  for (size_t i = 0; i < bins * bins * bins; i++) {
    m->SetBinContent(i, 1);
  }

  std::unique_ptr<TCollection> collection = std::make_unique<TObjArray>();
  collection->SetOwner(true);
  auto uni = std::make_unique<TF3>("uni", "1", 0, 1000000, 0, 1000000, 0, 1000000);
  auto* h = new TH3I("test", "test", bins, 0, 1000000, bins, 0, 1000000, bins, 0, 1000000);
  collection->Add(h);

  std::vector<Results> allResults;
  for (size_t r = 0; r < repetitions; r++) {
    h->Reset();
    h->FillRandom("uni", entries);
    Results iterationResults;
    iterationResults.sizeBytes = measure(Measurement::Size, m.get(), collection.get());
    iterationResults.sizeSerialisedBytes = measure(Measurement::SizeAfterSerialisation, m.get(), collection.get());
    iterationResults.deserialisationSeconds = measure(Measurement::Deserialisation, m.get(), collection.get());
    iterationResults.serialisationSeconds = measure(Measurement::Serialisation, m.get(), collection.get());
    iterationResults.mergingSeconds = measure(Measurement::Merging, m.get(), collection.get());
    allResults.push_back(iterationResults);
  }
  return allResults;
}

template <typename storageT>
static std::vector<Results> BM_BoostRegular1D(size_t repetitions, const Parameters p)
{
  const size_t entries = p.entries;
  const size_t bins = p.objectSize / sizeof(int32_t);
  const double min = 0.0;
  const double max = 1000000.0;

  auto merged = bh::make_histogram_with(storageT(), bh::axis::regular<>(bins, min, max, "x"));
  merged += merged; // avoid memory overcommitment by doing something with data.
  using HistoType = decltype(merged);

  TRandomMixMax gen;
  gen.SetSeed(std::random_device()());
  Double_t randomArray[entries];

  std::vector<Results> allResults;
  for (size_t r = 0; r < repetitions; r++) {
    auto h = bh::make_histogram_with(storageT(), bh::axis::regular<>(bins, min, max, "x"));
    gen.RndmArray(entries, randomArray);
    for (double rnd : randomArray) {
      h(rnd * max);
    }
    Results iterationResults;
    iterationResults.sizeBytes = measure(Measurement::Size, &merged, &h);
    iterationResults.sizeSerialisedBytes = measure(Measurement::SizeAfterSerialisation, &merged, &h);
    iterationResults.deserialisationSeconds = measure(Measurement::Deserialisation, &merged, &h);
    iterationResults.serialisationSeconds = measure(Measurement::Serialisation, &merged, &h);
    iterationResults.mergingSeconds = measure(Measurement::Merging, &merged, &h);
    allResults.push_back(iterationResults);
  }
  return allResults;
}

template <typename storageT>
static std::vector<Results> BM_BoostRegular2D(size_t repetitions, const Parameters p)
{
  const size_t entries = p.entries;
  const size_t bins = std::sqrt(p.objectSize / sizeof(int32_t));
  const double min = 0.0;
  const double max = 1000000.0;

  auto merged = bh::make_histogram_with(storageT(), bh::axis::regular<>(bins, min, max, "x"), bh::axis::regular<>(bins, min, max, "y"));
  merged += merged; // avoid memory overcommitment by doing something with data.

  TRandomMixMax gen;
  gen.SetSeed(std::random_device()());
  Double_t randomArrayX[entries];
  Double_t randomArrayY[entries];

  std::vector<Results> allResults;
  for (size_t r = 0; r < repetitions; r++) {
    auto h = bh::make_histogram_with(storageT(), bh::axis::regular<>(bins, min, max, "x"), bh::axis::regular<>(bins, min, max, "y"));
    gen.RndmArray(entries, randomArrayX);
    gen.RndmArray(entries, randomArrayY);
    for (size_t rnd = 0; rnd < entries; rnd++) {
      h(randomArrayX[rnd] * max, randomArrayY[rnd] * max);
    }

    Results iterationResults;
    iterationResults.sizeBytes = measure(Measurement::Size, &merged, &h);
    iterationResults.sizeSerialisedBytes = measure(Measurement::SizeAfterSerialisation, &merged, &h);
    iterationResults.deserialisationSeconds = measure(Measurement::Deserialisation, &merged, &h);
    iterationResults.serialisationSeconds = measure(Measurement::Serialisation, &merged, &h);
    iterationResults.mergingSeconds = measure(Measurement::Merging, &merged, &h);
    allResults.push_back(iterationResults);
  }
  return allResults;
}

static std::vector<Results> BM_THNSparseI(size_t repetitions, const Parameters p)
{
  const size_t bins = p.bins;
  const size_t dim = p.dimensions;
  const size_t entries = p.entries;

  const Double_t min = 0.0;
  const Double_t max = 1000000.0;
  const std::vector<Int_t> binsDims(dim, bins);
  const std::vector<Double_t> mins(dim, min);
  const std::vector<Double_t> maxs(dim, max);

  TRandomMixMax gen;
  gen.SetSeed(std::random_device()());
  Double_t randomArray[dim];

  std::vector<Results> allResults;
  for (size_t rep = 0; rep < repetitions; rep++) {
    // histograms have to be created in each loop repetition, otherwise i get strange segfaults with large number of entries.
    std::unique_ptr<TCollection> collection = std::make_unique<TObjArray>();
    collection->SetOwner(true);
    auto* h = new THnSparseI("test", "test", dim, binsDims.data(), mins.data(), maxs.data());
    collection->Add(h);

    auto m = std::make_unique<THnSparseI>("merged", "merged", dim, binsDims.data(), mins.data(), maxs.data());

    for (size_t entry = 0; entry < entries; entry++) {
      gen.RndmArray(dim, randomArray);
      for (double& r : randomArray) {
        r *= max;
      }
      h->Fill(randomArray);
    }

    for (size_t entry = 0; entry < entries; entry++) {
      gen.RndmArray(dim, randomArray);
      for (double& r : randomArray) {
        r *= max;
      }
      m->Fill(randomArray);
    }

    Results iterationResults;
    iterationResults.sizeBytes = measure(Measurement::Size, m.get(), collection.get());
    iterationResults.sizeSerialisedBytes = measure(Measurement::SizeAfterSerialisation, m.get(), collection.get());
    iterationResults.deserialisationSeconds = measure(Measurement::Deserialisation, m.get(), collection.get());
    iterationResults.serialisationSeconds = measure(Measurement::Serialisation, m.get(), collection.get());
    iterationResults.mergingSeconds = measure(Measurement::Merging, m.get(), collection.get());
    allResults.push_back(iterationResults);
  }
  return allResults;
}

static std::vector<Results> BM_TTree(size_t repetitions, const Parameters p)
{
  const size_t branchSize = p.branchSize;
  const size_t branches = p.branches;
  const size_t entries = p.entries;

  using branch_t = std::vector<uint64_t>;
  std::vector<branch_t> branchCollection;
  for (size_t i = 0; i < branches; i++) {
    branchCollection.emplace_back(branchSize, 0);
  }
  auto createTree = [&](std::string name) -> TTree* {
    TTree* tree = new TTree();
    for (size_t i = 0; i < branchCollection.size(); i++) {
      tree->Branch(("b" + std::to_string(i)).c_str(),
                   &branchCollection[i],
                   ("array" + std::to_string(i) + "[" + std::to_string(branchSize) + "]:l").c_str());
    }
    tree->SetName(name.c_str());
    return tree;
  };

  auto fillTree = [&](TTree* t) {
    TRandomMixMax gen;
    gen.SetSeed(std::random_device()());
    Float_t randomArray[branchSize];

    for (size_t entry = 0; entry < entries; entry++) {
      for (auto& branch : branchCollection) {
        gen.RndmArray(branchSize, randomArray);
        for (size_t i = 0; i < branchSize; i++) {
          branch[i] = static_cast<uint64_t>(randomArray[i]);
        }
      }
      t->Fill();
    }
  };

  std::vector<Results> allResults;
  for (size_t r = 0; r < repetitions; r++) {
    std::unique_ptr<TCollection> collection = std::make_unique<TObjArray>();
    collection->SetOwner(true);
    TTree* t = createTree("input");
    fillTree(t);
    collection->Add(t);

    TTree* m = createTree("merged");
    fillTree(m);

    Results iterationResults;
    iterationResults.sizeBytes = measure(Measurement::Size, m, collection.get());
    iterationResults.sizeSerialisedBytes = measure(Measurement::SizeAfterSerialisation, m, collection.get());
    iterationResults.deserialisationSeconds = measure(Measurement::Deserialisation, m, collection.get());
    iterationResults.serialisationSeconds = measure(Measurement::Serialisation, m, collection.get());
    iterationResults.mergingSeconds = measure(Measurement::Merging, m, collection.get());
    allResults.push_back(iterationResults);

    delete m;
  }
  return allResults;
}

void printHeaderCSV(std::ostream& out)
{
  out << "name,"
         "objectSize,bins,dimensions,entries,branches,branchSize,"
         "sizeBytes,sizeSerialisedBytes,deserialisationSeconds,mergingSeconds,serialisationSeconds"
         "\n";
}

void printResultsCSV(std::ostream& out, std::string name, const Parameters& p, const std::vector<Results>& results)
{
  for (const auto r : results) {
    out << name << ","
        << p.objectSize << "," << p.bins << "," << p.dimensions << "," << p.entries << "," << p.branches << "," << p.branchSize << ","
        << r.sizeBytes << "," << r.sizeSerialisedBytes << "," << r.deserialisationSeconds << "," << r.mergingSeconds << "," << r.serialisationSeconds
        << '\n';
  }
}

int main(int argc, const char* argv[])
{
  if (argc < 2) {
    throw std::runtime_error("Output file name expected");
  }

  std::ofstream file;
  file.open(argv[1]);
  printHeaderCSV(file);
  printHeaderCSV(std::cout);

  size_t repetitions = argc < 3 ? 1 : std::atoll(argv[2]);

  {
    // TH1I
    std::vector<Parameters> parameters{
      Parameters::forHistograms(8 << 0, 50000),
      Parameters::forHistograms(8 << 3, 50000),
      Parameters::forHistograms(8 << 6, 50000),
      Parameters::forHistograms(8 << 9, 50000),
      Parameters::forHistograms(8 << 12, 50000),
      Parameters::forHistograms(8 << 15, 50000),
      Parameters::forHistograms(8 << 18, 50000),
      Parameters::forHistograms(8 << 21, 50000)};
    for (const auto& p : parameters) {
      auto results = BM_TH1I(repetitions, p);
      printResultsCSV(file, "TH1I", p, results);
      printResultsCSV(std::cout, "TH1I", p, results);
    }
  }

  {
    // TH2I
    std::vector<Parameters> parameters{
      Parameters::forHistograms(8 << 0, 50000),
      Parameters::forHistograms(8 << 3, 50000),
      Parameters::forHistograms(8 << 6, 50000),
      Parameters::forHistograms(8 << 9, 50000),
      Parameters::forHistograms(8 << 12, 50000),
      Parameters::forHistograms(8 << 15, 50000),
      Parameters::forHistograms(8 << 18, 50000),
      Parameters::forHistograms(8 << 21, 50000)};
    for (const auto& p : parameters) {
      auto results = BM_TH2I(repetitions, p);
      printResultsCSV(file, "TH2I", p, results);
      printResultsCSV(std::cout, "TH2I", p, results);
    }
  }

  {
    // TH3I
    std::vector<Parameters> parameters{
      Parameters::forHistograms(8 << 0, 50000),
      Parameters::forHistograms(8 << 3, 50000),
      Parameters::forHistograms(8 << 6, 50000),
      Parameters::forHistograms(8 << 9, 50000),
      Parameters::forHistograms(8 << 12, 50000),
      Parameters::forHistograms(8 << 15, 50000),
      Parameters::forHistograms(8 << 18, 50000),
      Parameters::forHistograms(8 << 21, 50000)};
    for (const auto& p : parameters) {
      auto results = BM_TH3I(repetitions, p);
      printResultsCSV(file, "TH3I", p, results);
      printResultsCSV(std::cout, "TH3I", p, results);
    }
  }

  {
    // THnSparseI
    std::vector<Parameters> parameters{
      Parameters::forSparse(8, 8, 512),
      Parameters::forSparse(64, 8, 512),
      Parameters::forSparse(512, 8, 512),
      Parameters::forSparse(4096, 8, 512),
      Parameters::forSparse(32768, 8, 512),
      Parameters::forSparse(512, 2, 512),
      Parameters::forSparse(512, 4, 512),
      Parameters::forSparse(512, 8, 512),
      Parameters::forSparse(512, 16, 512),
      Parameters::forSparse(512, 32, 512),
      Parameters::forSparse(512, 64, 512),
      Parameters::forSparse(512, 8, 1),
      Parameters::forSparse(512, 8, 8),
      Parameters::forSparse(512, 8, 64),
      Parameters::forSparse(512, 8, 512),
      Parameters::forSparse(512, 8, 4096),
      Parameters::forSparse(512, 8, 32768),
      Parameters::forSparse(512, 8, 262144),
      Parameters::forSparse(512, 8, 2097152),
      //  Parameters::forSparse(512, 8, 16777216),
      Parameters::forSparse(32, 4, 1),
      Parameters::forSparse(32, 4, 8),
      Parameters::forSparse(32, 4, 64),
      Parameters::forSparse(32, 4, 512),
      Parameters::forSparse(32, 4, 4096),
      Parameters::forSparse(32, 4, 32768),
      Parameters::forSparse(32, 4, 262144),
      Parameters::forSparse(32, 4, 2097152),
      Parameters::forSparse(32, 4, 16777216)};
    for (const auto& p : parameters) {
      auto results = BM_THNSparseI(repetitions, p);
      printResultsCSV(file, "THnSparseI", p, results);
      printResultsCSV(std::cout, "THnSparseI", p, results);
    }
  }

  {
    // TTree
    std::vector<Parameters> parameters{
      Parameters::forTrees(8, 8, 8),
      Parameters::forTrees(8, 8, 8 << 3),
      Parameters::forTrees(8, 8, 8 << 6),
      Parameters::forTrees(8, 8, 8 << 9),
      Parameters::forTrees(8, 8, 8 << 12),
      Parameters::forTrees(8, 8, 8 << 15),
      Parameters::forTrees(8, 8, 8 << 18),
      Parameters::forTrees(8, 1 << 0, 8 << 12),
      Parameters::forTrees(8, 1 << 2, 8 << 12),
      Parameters::forTrees(8, 1 << 4, 8 << 12),
      Parameters::forTrees(8, 1 << 6, 8 << 12),
      Parameters::forTrees(8, 1 << 8, 8 << 12),
      Parameters::forTrees(1 << 0, 8, 8 << 12),
      Parameters::forTrees(1 << 2, 8, 8 << 12),
      Parameters::forTrees(1 << 4, 8, 8 << 12),
      Parameters::forTrees(1 << 6, 8, 8 << 12),
      Parameters::forTrees(1 << 8, 8, 8 << 12)};
    for (const auto& p : parameters) {
      auto results = BM_TTree(repetitions, p);
      printResultsCSV(file, "TTree", p, results);
      printResultsCSV(std::cout, "TTree", p, results);
    }
  }

  {
    // boost regular 1D. We use a combination of template and macro to be able to use static storage (std::array) with different parameters.
#define BM_BOOST1DARRAY_FOR(objSize, entries)                                                                  \
  {                                                                                                            \
    constexpr auto p = Parameters::forHistograms(objSize, entries);                                            \
    auto results = BM_BoostRegular1D<std::array<int32_t, p.objectSize / sizeof(int32_t) + 2>>(repetitions, p); \
    printResultsCSV(file, "BoostRegular1DArray", p, results);                                                  \
    printResultsCSV(std::cout, "BoostRegular1DArray", p, results);                                             \
  }

    BM_BOOST1DARRAY_FOR(8 << 0, 50000);
    BM_BOOST1DARRAY_FOR(8 << 3, 50000);
    BM_BOOST1DARRAY_FOR(8 << 6, 50000);
    BM_BOOST1DARRAY_FOR(8 << 9, 50000);
    BM_BOOST1DARRAY_FOR(8 << 12, 50000);
    BM_BOOST1DARRAY_FOR(8 << 15, 50000);
  }

  {
    // boost regular 1D.
#define BM_BOOST1DVECTOR_FOR(objSize, entries)                              \
  {                                                                         \
    constexpr auto p = Parameters::forHistograms(objSize, entries);         \
    auto results = BM_BoostRegular1D<std::vector<int32_t>>(repetitions, p); \
    printResultsCSV(file, "BoostRegular1DVector", p, results);              \
    printResultsCSV(std::cout, "BoostRegular1DVector", p, results);         \
  }

    BM_BOOST1DVECTOR_FOR(8 << 0, 50000);
    BM_BOOST1DVECTOR_FOR(8 << 3, 50000);
    BM_BOOST1DVECTOR_FOR(8 << 6, 50000);
    BM_BOOST1DVECTOR_FOR(8 << 9, 50000);
    BM_BOOST1DVECTOR_FOR(8 << 12, 50000);
    BM_BOOST1DVECTOR_FOR(8 << 15, 50000);
    BM_BOOST1DVECTOR_FOR(8 << 18, 50000);
    BM_BOOST1DVECTOR_FOR(8 << 21, 50000);
  }

  {
    // boost regular 2D. We use a combination of template and macro to be able to use static storage (std::array) with different parameters.
#define BM_BOOST2DARRAY_FOR(objSize, arrSize, entries)                              \
  {                                                                                 \
    constexpr auto p = Parameters::forHistograms(objSize, entries);                 \
    auto results = BM_BoostRegular2D<std::array<int32_t, arrSize>>(repetitions, p); \
    printResultsCSV(file, "BoostRegular2DArray", p, results);                       \
    printResultsCSV(std::cout, "BoostRegular2DArray", p, results);                  \
  }

    BM_BOOST2DARRAY_FOR(8 << 0, 10, 50000);
    BM_BOOST2DARRAY_FOR(8 << 3, 36, 50000);
    BM_BOOST2DARRAY_FOR(8 << 6, 178, 50000);
    BM_BOOST2DARRAY_FOR(8 << 9, 1156, 50000);
    BM_BOOST2DARRAY_FOR(8 << 12, 8558, 50000);
    BM_BOOST2DARRAY_FOR(8 << 15, 66564, 50000);
  }

  {
    // boost regular 2D.
#define BM_BOOST2DVECTOR_FOR(objSize, entries)                              \
  {                                                                         \
    constexpr auto p = Parameters::forHistograms(objSize, entries);         \
    auto results = BM_BoostRegular2D<std::vector<int32_t>>(repetitions, p); \
    printResultsCSV(file, "BoostRegular2DVector", p, results);              \
    printResultsCSV(std::cout, "BoostRegular2DVector", p, results);         \
  }

    BM_BOOST2DVECTOR_FOR(8 << 0, 50000);
    BM_BOOST2DVECTOR_FOR(8 << 3, 50000);
    BM_BOOST2DVECTOR_FOR(8 << 6, 50000);
    BM_BOOST2DVECTOR_FOR(8 << 9, 50000);
    BM_BOOST2DVECTOR_FOR(8 << 12, 50000);
    BM_BOOST2DVECTOR_FOR(8 << 15, 50000);
    BM_BOOST2DVECTOR_FOR(8 << 18, 50000);
    BM_BOOST2DVECTOR_FOR(8 << 21, 50000);
  }

  file.close();
  return 0;
}
