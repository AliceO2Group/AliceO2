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

#ifndef FRAMEWORK_HISTOGRAMREGISTRY_H_
#define FRAMEWORK_HISTOGRAMREGISTRY_H_

#include "Framework/HistogramSpec.h"
#include "Framework/ASoA.h"
#include "Framework/FunctionalHelpers.h"
#include "Framework/Logger.h"
#include "Framework/OutputRef.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/OutputSpec.h"
#include "Framework/SerializationMethods.h"
#include "Framework/TableBuilder.h"
#include "Framework/RuntimeError.h"

#include <TDataMember.h>
#include <TDataType.h>

#include <deque>

class TList;

#define HIST(name) CONST_STR(name)

namespace o2::framework
{
//**************************************************************************************************
/**
 * Static helper class to fill root histograms of any type. Contains functionality to fill once per call or a whole (filtered) table at once.
 */
//**************************************************************************************************
struct HistFiller {
  // fill any type of histogram (if weight was requested it must be the last argument)
  template <typename T, typename... Ts>
  static void fillHistAny(std::shared_ptr<T> hist, const Ts&... positionAndWeight);

  // fill any type of histogram with columns (Cs) of a filtered table (if weight is requested it must reside the last specified column)
  template <typename... Cs, typename R, typename T>
  static void fillHistAny(std::shared_ptr<R> hist, const T& table, const o2::framework::expressions::Filter& filter);

  // function that returns rough estimate for the size of a histogram in MB
  template <typename T>
  static double getSize(std::shared_ptr<T> hist, double fillFraction = 1.);

 private:
  // helper function to determine base element size of histograms (in bytes)
  template <typename T>
  static int getBaseElementSize(T* ptr);

  template <typename T, typename Next, typename... Rest, typename P>
  static int getBaseElementSize(P* ptr);

  template <typename B, typename T>
  static int getBaseElementSize(T* ptr);
};

//**************************************************************************************************
/**
 * HistogramRegistry for storing and filling histograms of any type.
 */
//**************************************************************************************************
class HistogramRegistry
{
  // HistogramName class providing the associated hash and a first guess for the index in the registry
  struct HistName {
    // ctor for histogram names that are already hashed at compile time via HIST("myHistName")
    template <char... chars>
    constexpr HistName(const ConstStr<chars...>& hashedHistName);
    char const* const str{};
    const uint32_t hash{};
    const uint32_t idx{};

   protected:
    friend class HistogramRegistry;
    // ctor that does the hashing at runtime (for internal use only)
    constexpr HistName(char const* const name);
  };

 public:
  HistogramRegistry(char const* const name = "histograms", std::vector<HistogramSpec> histSpecs = {}, OutputObjHandlingPolicy policy = OutputObjHandlingPolicy::AnalysisObject, bool sortHistos = false, bool createRegistryDir = false);

  // functions to add histograms to the registry
  HistPtr add(const HistogramSpec& histSpec);
  HistPtr add(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2 = false);
  HistPtr add(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2 = false);
  HistPtr add(const std::string& name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2 = false);

  template <typename T>
  std::shared_ptr<T> add(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2 = false);
  template <typename T>
  std::shared_ptr<T> add(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2 = false);

  void addClone(const std::string& source, const std::string& target);

  // function to query if name is already in use
  bool contains(const HistName& histName);

  // get the underlying histogram pointer
  template <typename T>
  std::shared_ptr<T> get(const HistName& histName);

  template <typename T>
  std::shared_ptr<T> operator()(const HistName& histName);

  // return the OutputSpec associated to the HistogramRegistry
  OutputSpec const spec();

  OutputRef ref();

  void setHash(uint32_t hash);

  TList* operator*();

  // fill hist with values
  template <typename... Ts>
  void fill(const HistName& histName, Ts&&... positionAndWeight);

  // fill hist with content of (filtered) table columns
  template <typename... Cs, typename T>
  void fill(const HistName& histName, const T& table, const o2::framework::expressions::Filter& filter);

  // get rough estimate for size of histogram stored in registry
  double getSize(const HistName& histName, double fillFraction = 1.);

  // get rough estimate for size of all histograms stored in registry
  double getSize(double fillFraction = 1.);

  // print summary of the histograms stored in registry
  void print(bool showAxisDetails = false);

  // lookup distance counter for benchmarking
  mutable uint32_t lookup = 0;

 private:
  // create histogram from specification and insert it into the registry
  HistPtr insert(const HistogramSpec& histSpec);

  // clone an existing histogram and insert it into the registry
  template <typename T>
  HistPtr insertClone(const HistName& histName, const std::shared_ptr<T> originalHist);

  // helper function that checks if histogram name can be used in registry
  void validateHistName(const std::string& name, const uint32_t hash);

  // helper function to find the histogram position in the registry
  template <typename T>
  uint32_t getHistIndex(const T& histName);

  constexpr uint32_t imask(uint32_t i) const
  {
    return i & REGISTRY_BITMASK;
  }

  // helper function to create resp. find the subList defined by path
  TList* getSubList(TList* list, std::deque<std::string>& path);

  // helper function to split user defined path/to/hist/name string
  std::deque<std::string> splitPath(const std::string& pathAndNameUser);

  // helper function that checks if name of histogram is reasonable and keeps track of names already in use
  void registerName(const std::string& name);

  std::string mName{};
  OutputObjHandlingPolicy mPolicy{};
  bool mCreateRegistryDir{};
  bool mSortHistos{};
  uint32_t mTaskHash{};
  std::vector<std::string> mRegisteredNames{};

  // the maximum number of histograms in buffer is currently set to 512
  // which seems to be both reasonably large and allowing for very fast lookup
  static constexpr uint32_t REGISTRY_BITMASK{0x1FF};
  static constexpr uint32_t MAX_REGISTRY_SIZE{REGISTRY_BITMASK + 1};
  std::array<uint32_t, MAX_REGISTRY_SIZE> mRegistryKey{};
  std::array<HistPtr, MAX_REGISTRY_SIZE> mRegistryValue{};
};

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
// Implementation of HistFiller template functions.
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------

template <typename T, typename... Ts>
void HistFiller::fillHistAny(std::shared_ptr<T> hist, const Ts&... positionAndWeight)
{
  constexpr int nArgs = sizeof...(Ts);

  constexpr bool validTH3 = (std::is_same_v<TH3, T> && (nArgs == 3 || nArgs == 4));
  constexpr bool validTH2 = (std::is_same_v<TH2, T> && (nArgs == 2 || nArgs == 3));
  constexpr bool validTH1 = (std::is_same_v<TH1, T> && (nArgs == 1 || nArgs == 2));
  constexpr bool validTProfile3D = (std::is_same_v<TProfile3D, T> && (nArgs == 4 || nArgs == 5));
  constexpr bool validTProfile2D = (std::is_same_v<TProfile2D, T> && (nArgs == 3 || nArgs == 4));
  constexpr bool validTProfile = (std::is_same_v<TProfile, T> && (nArgs == 2 || nArgs == 3));

  constexpr bool validSimpleFill = validTH1 || validTH2 || validTH3 || validTProfile || validTProfile2D || validTProfile3D;
  // unfortunately we dont know at compile the dimension of THn(Sparse)
  constexpr bool validComplexFill = std::is_base_of_v<THnBase, T>;
  constexpr bool validComplexFillStep = std::is_base_of_v<StepTHn, T>;

  if constexpr (validSimpleFill) {
    hist->Fill(static_cast<double>(positionAndWeight)...);
  } else if constexpr (validComplexFillStep) {
    hist->Fill(positionAndWeight...); // first argument in pack is iStep, dimension check is done in StepTHn itself
  } else if constexpr (validComplexFill) {
    double tempArray[] = {static_cast<double>(positionAndWeight)...};
    double weight{1.};
    constexpr int nArgsMinusOne = nArgs - 1;
    if (hist->GetNdimensions() == nArgsMinusOne) {
      weight = tempArray[nArgsMinusOne];
    } else if (hist->GetNdimensions() != nArgs) {
      LOGF(fatal, "The number of arguments in fill function called for histogram %s is incompatible with histogram dimensions.", hist->GetName());
    }
    hist->Fill(tempArray, weight);
  } else {
    LOGF(fatal, "The number of arguments in fill function called for histogram %s is incompatible with histogram dimensions.", hist->GetName());
  }
}

template <typename... Cs, typename R, typename T>
void HistFiller::fillHistAny(std::shared_ptr<R> hist, const T& table, const o2::framework::expressions::Filter& filter)
{
  if constexpr (std::is_base_of_v<StepTHn, T>) {
    LOGF(fatal, "Table filling is not (yet?) supported for StepTHn.");
    return;
  }
  auto s = o2::framework::expressions::createSelection(table.asArrowTable(), filter);
  auto filtered = o2::soa::Filtered<T>{{table.asArrowTable()}, s};
  for (auto& t : filtered) {
    fillHistAny(hist, (*(static_cast<Cs>(t).getIterator()))...);
  }
}

template <typename T>
double HistFiller::getSize(std::shared_ptr<T> hist, double fillFraction)
{
  double size{0.};
  if constexpr (std::is_base_of_v<TH1, T>) {
    size = hist->GetNcells() * (HistFiller::getBaseElementSize(hist.get()) + ((hist->GetSumw2()->fN) ? sizeof(double) : 0.));
  } else if constexpr (std::is_base_of_v<THn, T>) {
    size = hist->GetNbins() * (HistFiller::getBaseElementSize(hist.get()) + ((hist->GetSumw2() != -1.) ? sizeof(double) : 0.));
  } else if constexpr (std::is_base_of_v<THnSparse, T>) {
    // THnSparse has massive overhead and should only be used when histogram is large and a very small fraction of bins is filled
    double nBinsTotal = 1.;
    int compCoordSize = 0; // size required to store a compact coordinate representation
    for (int d = 0; d < hist->GetNdimensions(); ++d) {
      int nBins = hist->GetAxis(d)->GetNbins() + 2;
      nBinsTotal *= nBins;

      // number of bits needed to store compact coordinates
      int b = 1;
      while (nBins /= 2) {
        ++b;
      }
      compCoordSize += b;
    }
    compCoordSize = (compCoordSize + 7) / 8; // turn bits into bytes

    // THnSparse stores the data in an array of chunks (THnSparseArrayChunk), each containing a fixed number of bins (e.g. 1024 * 16)
    double nBinsFilled = fillFraction * nBinsTotal;
    int nCunks = ceil(nBinsFilled / hist->GetChunkSize());
    int chunkOverhead = sizeof(THnSparseArrayChunk);

    // each chunk holds array of compact bin-coordinates and an array of bin content (+ one of bin error if requested)
    double binSize = compCoordSize + HistFiller::getBaseElementSize(hist.get()) + ((hist->GetSumw2() != -1.) ? sizeof(double) : 0.);
    size = nCunks * (chunkOverhead + hist->GetChunkSize() * binSize);
    // since THnSparse must keep track of all the stored bins, it stores a map that
    // relates the compact bin coordinates (or a hash thereof) to a linear index
    // this index determines in which chunk and therein at which position to find / store bin coordinate and content
    size += nBinsFilled * 3 * sizeof(Long64_t); // hash, key, value; not sure why 3 are needed here...
  }
  return size / 1048576.;
}

template <typename T>
int HistFiller::getBaseElementSize(T* ptr)
{
  if constexpr (std::is_base_of_v<TH1, T> || std::is_base_of_v<THnSparse, T>) {
    return getBaseElementSize<TArrayD, TArrayF, TArrayC, TArrayI, TArrayC, TArrayL>(ptr);
  } else {
    return getBaseElementSize<double, float, int, short, char, long>(ptr);
  }
}

template <typename T, typename Next, typename... Rest, typename P>
int HistFiller::getBaseElementSize(P* ptr)
{
  if (auto size = getBaseElementSize<T>(ptr)) {
    return size;
  }
  return getBaseElementSize<Next, Rest...>(ptr);
}

template <typename B, typename T>
int HistFiller::getBaseElementSize(T* ptr)
{
  if constexpr (std::is_base_of_v<THn, T>) {
    if (dynamic_cast<THnT<B>*>(ptr)) {
      return sizeof(B);
    }
  } else if constexpr (std::is_base_of_v<THnSparse, T>) {
    if (dynamic_cast<THnSparseT<B>*>(ptr)) {
      TDataMember* dm = B::Class()->GetDataMember("fArray");
      return dm ? dm->GetDataType()->Size() : 0;
    }
  } else if constexpr (std::is_base_of_v<TH1, T>) {
    if (auto arrayPtr = dynamic_cast<B*>(ptr)) {
      return sizeof(arrayPtr->At(0));
    }
  }
  return 0;
}

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
// Implementation of HistogramRegistry template functions.
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------

template <char... chars>
constexpr HistogramRegistry::HistName::HistName(const ConstStr<chars...>& hashedHistName)
  : str(hashedHistName.str),
    hash(hashedHistName.hash),
    idx(hash & REGISTRY_BITMASK)
{
}

template <typename T>
std::shared_ptr<T> HistogramRegistry::add(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2)
{
  auto histVariant = add(name, title, histConfigSpec, callSumw2);
  if (auto histPtr = std::get_if<std::shared_ptr<T>>(&histVariant)) {
    return *histPtr;
  } else {
    throw runtime_error_f(R"(Histogram type specified in add<>("%s") does not match the actual type of the histogram!)", name);
  }
}

template <typename T>
std::shared_ptr<T> HistogramRegistry::add(char const* const name, char const* const title, HistType histType, const std::vector<AxisSpec>& axes, bool callSumw2)
{
  auto histVariant = add(name, title, histType, axes, callSumw2);
  if (auto histPtr = std::get_if<std::shared_ptr<T>>(&histVariant)) {
    return *histPtr;
  } else {
    throw runtime_error_f(R"(Histogram type specified in add<>("%s") does not match the actual type of the histogram!)", name);
  }
}

template <typename T>
std::shared_ptr<T> HistogramRegistry::get(const HistName& histName)
{
  if (auto histPtr = std::get_if<std::shared_ptr<T>>(&mRegistryValue[getHistIndex(histName)])) {
    return *histPtr;
  } else {
    throw runtime_error_f(R"(Histogram type specified in get<>(HIST("%s")) does not match the actual type of the histogram!)", histName.str);
  }
}

template <typename T>
std::shared_ptr<T> HistogramRegistry::operator()(const HistName& histName)
{
  return get<T>(histName);
}

template <typename T>
HistPtr HistogramRegistry::insertClone(const HistName& histName, const std::shared_ptr<T> originalHist)
{
  validateHistName(histName.str, histName.hash);
  for (auto i = 0u; i < MAX_REGISTRY_SIZE; ++i) {
    TObject* rawPtr = nullptr;
    std::visit([&](const auto& sharedPtr) { rawPtr = sharedPtr.get(); }, mRegistryValue[imask(histName.idx + i)]);
    if (!rawPtr) {
      registerName(histName.str);
      mRegistryKey[imask(histName.idx + i)] = histName.hash;
      mRegistryValue[imask(histName.idx + i)] = std::shared_ptr<T>(static_cast<T*>(originalHist->Clone(histName.str)));
      lookup += i;
      return mRegistryValue[imask(histName.idx + i)];
    }
  }
  LOGF(fatal, R"(Internal array of HistogramRegistry "%s" is full.)", mName);
  return HistPtr();
}

template <typename T>
uint32_t HistogramRegistry::getHistIndex(const T& histName)
{
  if (O2_BUILTIN_LIKELY(histName.hash == mRegistryKey[histName.idx])) {
    return histName.idx;
  }
  for (auto i = 1u; i < MAX_REGISTRY_SIZE; ++i) {
    if (histName.hash == mRegistryKey[imask(histName.idx + i)]) {
      return imask(histName.idx + i);
    }
  }
  throw runtime_error_f(R"(Could not find histogram "%s" in HistogramRegistry "%s"!)", histName.str, mName.data());
}

template <typename... Ts>
void HistogramRegistry::fill(const HistName& histName, Ts&&... positionAndWeight)
{
  std::visit([&positionAndWeight...](auto&& hist) { HistFiller::fillHistAny(hist, std::forward<Ts>(positionAndWeight)...); }, mRegistryValue[getHistIndex(histName)]);
}

template <typename... Cs, typename T>
void HistogramRegistry::fill(const HistName& histName, const T& table, const o2::framework::expressions::Filter& filter)
{
  std::visit([&table, &filter](auto&& hist) { HistFiller::fillHistAny<Cs...>(hist, table, filter); }, mRegistryValue[getHistIndex(histName)]);
}

} // namespace o2::framework
#endif // FRAMEWORK_HISTOGRAMREGISTRY_H_
