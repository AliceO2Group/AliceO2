// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_HISTOGRAMREGISTRY_H_
#define FRAMEWORK_HISTOGRAMREGISTRY_H_

#include "Framework/ASoA.h"
#include "Framework/FunctionalHelpers.h"
#include "Framework/Logger.h"
#include "Framework/OutputRef.h"
#include "Framework/OutputObjHeader.h"
#include "Framework/OutputSpec.h"
#include "Framework/SerializationMethods.h"
#include "Framework/StringHelpers.h"
#include "Framework/TableBuilder.h"
#include "Framework/RuntimeError.h"

#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THn.h>
#include <THnSparse.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TProfile3D.h>
#include <TList.h>
#include <TDataMember.h>
#include <TDataType.h>

#include <string>
#include <variant>
#include <deque>

namespace o2::framework
{
// Available root histogram types
enum HistType : unsigned int {
  kUndefinedHist = 0,
  kTH1D,
  kTH1F,
  kTH1I,
  kTH1C,
  kTH1S,
  kTH2D,
  kTH2F,
  kTH2I,
  kTH2C,
  kTH2S,
  kTH3D,
  kTH3F,
  kTH3I,
  kTH3C,
  kTH3S,
  kTHnD,
  kTHnF,
  kTHnI,
  kTHnC,
  kTHnS,
  kTHnL,
  kTHnSparseD,
  kTHnSparseF,
  kTHnSparseI,
  kTHnSparseC,
  kTHnSparseS,
  kTHnSparseL,
  kTProfile,
  kTProfile2D,
  kTProfile3D,
  kStepTHnF, // FIXME: for these two to work we need to align StepTHn ctors with the root THn ones
  kStepTHnD
};

// variant of all possible root pointers; here we use only the interface types since the underlying data representation (int,float,double,long,char) is irrelevant
using HistPtr = std::variant<std::shared_ptr<THn>, std::shared_ptr<THnSparse>, std::shared_ptr<TH3>, std::shared_ptr<TH2>, std::shared_ptr<TH1>, std::shared_ptr<TProfile3D>, std::shared_ptr<TProfile2D>, std::shared_ptr<TProfile>>;

//**************************************************************************************************
/**
 * Specification of an Axis.
 */
//**************************************************************************************************
struct AxisSpec {
  AxisSpec(std::vector<double> binEdges_, std::optional<std::string> title_ = std::nullopt, std::optional<std::string> name_ = std::nullopt)
    : nBins(std::nullopt),
      binEdges(binEdges_),
      title(title_),
      name(name_)
  {
  }

  AxisSpec(int nBins_, double binMin_, double binMax_, std::optional<std::string> title_ = std::nullopt, std::optional<std::string> name_ = std::nullopt)
    : nBins(nBins_),
      binEdges({binMin_, binMax_}),
      title(title_),
      name(name_)
  {
  }

  std::optional<int> nBins{};
  std::vector<double> binEdges{};
  std::optional<std::string> title{};
  std::optional<std::string> name{}; // optional axis name for ndim histograms
};

//**************************************************************************************************
/**
 * Specification of a histogram configuration.
 */
//**************************************************************************************************
struct HistogramConfigSpec {
  HistogramConfigSpec(HistType type_, std::vector<AxisSpec> axes_)
    : type(type_),
      axes(axes_)
  {
  }
  HistogramConfigSpec() = default;
  HistogramConfigSpec(HistogramConfigSpec const& other) = default;
  HistogramConfigSpec(HistogramConfigSpec&& other) = default;

  void addAxis(const AxisSpec& axis)
  {
    axes.push_back(axis);
  }

  void addAxis(int nBins_, double binMin_, double binMax_, std::optional<std::string> title_ = std::nullopt, std::optional<std::string> name_ = std::nullopt)
  {
    axes.push_back({nBins_, binMin_, binMax_, title_, name_});
  }

  void addAxis(std::vector<double> binEdges_, std::optional<std::string> title_ = std::nullopt, std::optional<std::string> name_ = std::nullopt)
  {
    axes.push_back({binEdges_, title_, name_});
  }

  void addAxes(std::vector<AxisSpec> axes_)
  {
    axes.insert(axes.end(), axes_.begin(), axes_.end());
  }

  // add axes defined in other HistogramConfigSpec object
  void addAxes(const HistogramConfigSpec& other)
  {
    axes.insert(axes.end(), other.axes.begin(), other.axes.end());
  }

  HistType type{HistType::kUndefinedHist};
  std::vector<AxisSpec> axes{};
};

//**************************************************************************************************
/**
 * Specification of a histogram.
 */
//**************************************************************************************************
struct HistogramSpec {
  HistogramSpec(char const* const name_, char const* const title_, HistogramConfigSpec config_, bool callSumw2_ = false)
    : name(name_),
      id(compile_time_hash(name_)),
      title(title_),
      config(config_),
      callSumw2(callSumw2_)
  {
  }

  HistogramSpec()
    : name(""),
      id(0),
      config()
  {
  }
  HistogramSpec(HistogramSpec const& other) = default;
  HistogramSpec(HistogramSpec&& other) = default;

  std::string name{};
  uint32_t id{};
  std::string title{};
  HistogramConfigSpec config{};
  bool callSumw2{}; // wether or not hist needs heavy error structure produced by Sumw2()
};

//**************************************************************************************************
/**
 * Static helper class to generate histograms from the specifications.
 * Also provides functions to obtain pointer to the created histogram casted to the correct alternative of the std::variant HistPtr that is used in HistogramRegistry.
 */
//**************************************************************************************************
struct HistFactory {

  // create histogram of type T with the axes defined in HistogramSpec
  template <typename T>
  static std::shared_ptr<T> createHist(const HistogramSpec& histSpec)
  {
    constexpr std::size_t MAX_DIM{10};
    const std::size_t nAxes{histSpec.config.axes.size()};
    if (nAxes == 0 || nAxes > MAX_DIM) {
      LOGF(FATAL, "The histogram specification contains no (or too many) axes.");
      return nullptr;
    }

    int nBins[MAX_DIM]{0};
    double lowerBounds[MAX_DIM]{0.};
    double upperBounds[MAX_DIM]{0.};

    // first figure out number of bins and dimensions
    for (std::size_t i = 0; i < nAxes; i++) {
      nBins[i] = (histSpec.config.axes[i].nBins) ? *histSpec.config.axes[i].nBins : histSpec.config.axes[i].binEdges.size() - 1;
      lowerBounds[i] = histSpec.config.axes[i].binEdges.front();
      upperBounds[i] = histSpec.config.axes[i].binEdges.back();
    }

    // create histogram
    std::shared_ptr<T> hist{generateHist<T>(histSpec.name, histSpec.title, nAxes, nBins, lowerBounds, upperBounds)};
    if (!hist) {
      LOGF(FATAL, "The number of dimensions specified for histogram %s does not match the type.", histSpec.name);
      return nullptr;
    }

    // set axis properties
    for (std::size_t i = 0; i < nAxes; i++) {
      TAxis* axis{getAxis(i, hist)};
      if (axis) {
        if (histSpec.config.axes[i].title) {
          axis->SetTitle((*histSpec.config.axes[i].title).data());
        }

        // this helps to have axes not only called 0,1,2... in ndim histos
        if constexpr (std::is_base_of_v<THnBase, T>) {
          if (histSpec.config.axes[i].name) {
            axis->SetName((std::string(axis->GetName()) + "-" + *histSpec.config.axes[i].name).data());
          }
        }

        // move the bin edges in case a variable binning was requested
        if (!histSpec.config.axes[i].nBins) {
          if (!std::is_sorted(std::begin(histSpec.config.axes[i].binEdges), std::end(histSpec.config.axes[i].binEdges))) {
            LOGF(FATAL, "The bin edges in histogram %s are not in increasing order!", histSpec.name);
            return nullptr;
          }
          axis->Set(nBins[i], histSpec.config.axes[i].binEdges.data());
        }
      }
    }
    if (histSpec.callSumw2) {
      hist->Sumw2();
    }

    return hist;
  }

  // create histogram and return it casted to the correct alternative held in HistPtr variant
  template <typename T>
  static HistPtr createHistVariant(const HistogramSpec& histSpec)
  {
    if (auto hist = castToVariant(createHist<T>(histSpec))) {
      return *hist;
    } else {
      throw runtime_error("Histogram was not created properly.");
    }
  }

  // runtime version of the above
  static HistPtr createHistVariant(const HistogramSpec& histSpec)
  {
    if (histSpec.config.type == HistType::kUndefinedHist) {
      throw runtime_error("Histogram type was not specified.");
    } else {
      return HistogramCreationCallbacks.at(histSpec.config.type)(histSpec);
    }
  }

  // helper function to get the axis via index for any type of root histogram
  template <typename T>
  static TAxis* getAxis(const int i, std::shared_ptr<T>& hist)
  {
    if constexpr (std::is_base_of_v<THnBase, T>) {
      return hist->GetAxis(i);
    } else {
      return (i == 0) ? hist->GetXaxis()
                      : (i == 1) ? hist->GetYaxis()
                                 : (i == 2) ? hist->GetZaxis()
                                            : nullptr;
    }
  }

 private:
  static const std::map<HistType, std::function<HistPtr(const HistogramSpec&)>> HistogramCreationCallbacks;

  // helper function to generate the actual histograms
  template <typename T>
  static T* generateHist(const std::string& name, const std::string& title, const std::size_t nDim,
                         const int nBins[], const double lowerBounds[], const double upperBounds[])
  {
    if constexpr (std::is_base_of_v<THnBase, T>) {
      return new T(name.data(), title.data(), nDim, nBins, lowerBounds, upperBounds);
    } else if constexpr (std::is_base_of_v<TH3, T>) {
      return (nDim != 3) ? nullptr
                         : new T(name.data(), title.data(), nBins[0], lowerBounds[0],
                                 upperBounds[0], nBins[1], lowerBounds[1], upperBounds[1],
                                 nBins[2], lowerBounds[2], upperBounds[2]);
    } else if constexpr (std::is_base_of_v<TH2, T>) {
      return (nDim != 2) ? nullptr
                         : new T(name.data(), title.data(), nBins[0], lowerBounds[0],
                                 upperBounds[0], nBins[1], lowerBounds[1], upperBounds[1]);
    } else if constexpr (std::is_base_of_v<TH1, T>) {
      return (nDim != 1)
               ? nullptr
               : new T(name.data(), title.data(), nBins[0], lowerBounds[0], upperBounds[0]);
    }
    return nullptr;
  }

  // helper function to cast the actual histogram type (e.g. TH2F) to the correct interface type (e.g. TH2) that is stored in the HistPtr variant
  template <typename T>
  static std::optional<HistPtr> castToVariant(std::shared_ptr<TObject> obj)
  {
    if (obj->InheritsFrom(T::Class())) {
      return std::static_pointer_cast<T>(obj);
    }
    return std::nullopt;
  }

  template <typename T, typename Next, typename... Rest>
  static std::optional<HistPtr> castToVariant(std::shared_ptr<TObject> obj)
  {
    if (auto hist = castToVariant<T>(obj)) {
      return hist;
    }
    return castToVariant<Next, Rest...>(obj);
  }

  static std::optional<HistPtr> castToVariant(std::shared_ptr<TObject> obj)
  {
    if (obj) {
      // TProfile3D is TH3, TProfile2D is TH2, TH3 is TH1, TH2 is TH1, TProfile is TH1
      return castToVariant<THn, THnSparse, TProfile3D, TH3, TProfile2D, TH2, TProfile, TH1>(obj);
    }
    return std::nullopt;
  }
};

//**************************************************************************************************
/**
 * Static helper class to fill existing root histograms of any type.
 * Contains functionality to fill once per call or a whole (filtered) table at once.
 */
//**************************************************************************************************
struct HistFiller {
  // fill any type of histogram (if weight was requested it must be the last argument)
  template <typename T, typename... Ts>
  static void fillHistAny(std::shared_ptr<T>& hist, const Ts&... positionAndWeight)
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

    if constexpr (validSimpleFill) {
      hist->Fill(static_cast<double>(positionAndWeight)...);
    } else if constexpr (validComplexFill) {
      double tempArray[] = {static_cast<double>(positionAndWeight)...};
      double weight{1.};
      if (hist->GetNdimensions() == nArgs - 1) {
        weight = tempArray[nArgs - 1];
      } else if (hist->GetNdimensions() != nArgs) {
        LOGF(FATAL, "The number of arguments in fill function called for histogram %s is incompatible with histogram dimensions.", hist->GetName());
      }
      hist->Fill(tempArray, weight);
    } else {
      LOGF(FATAL, "The number of arguments in fill function called for histogram %s is incompatible with histogram dimensions.", hist->GetName());
    }
  }

  // fill any type of histogram with columns (Cs) of a filtered table (if weight is requested it must reside the last specified column)
  template <typename... Cs, typename R, typename T>
  static void fillHistAny(std::shared_ptr<R>& hist, const T& table, const o2::framework::expressions::Filter& filter)
  {
    auto filtered = o2::soa::Filtered<T>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)};
    for (auto& t : filtered) {
      fillHistAny(hist, (*(static_cast<Cs>(t).getIterator()))...);
    }
  }

  // function that returns rough estimate for the size of a histogram in MB
  template <typename T>
  static double getSize(std::shared_ptr<T>& hist, double fillFraction = 1.)
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

 private:
  // helper function to determine base element size of histograms (in bytes)
  // the complicated casting gymnastics are needed here since we only store the interface types in the registry
  template <typename T>
  static int getBaseElementSize(T* ptr)
  {
    if constexpr (std::is_base_of_v<TH1, T> || std::is_base_of_v<THnSparse, T>) {
      return getBaseElementSize<TArrayD, TArrayF, TArrayC, TArrayI, TArrayC, TArrayL>(ptr);
    } else {
      return getBaseElementSize<double, float, int, short, char, long>(ptr);
    }
  }

  template <typename T, typename Next, typename... Rest, typename P>
  static int getBaseElementSize(P* ptr)
  {
    if (auto size = getBaseElementSize<T>(ptr)) {
      return size;
    }
    return getBaseElementSize<Next, Rest...>(ptr);
  }

  template <typename B, typename T>
  static int getBaseElementSize(T* ptr)
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
  };
};

//**************************************************************************************************
/**
 * Histogram registry that can be used to store and fill histograms of any type.
 */
//**************************************************************************************************
class HistogramRegistry
{
 public:
  HistogramRegistry(char const* const name, std::vector<HistogramSpec> histSpecs = {}, OutputObjHandlingPolicy policy = OutputObjHandlingPolicy::AnalysisObject, bool sortHistos = true, bool createRegistryDir = false)
    : mName(name), mPolicy(policy), mRegistryKey(), mRegistryValue(), mSortHistos(sortHistos), mCreateRegistryDir(createRegistryDir)
  {
    mRegistryKey.fill(0u);
    for (auto& histSpec : histSpecs) {
      insert(histSpec);
    }
  }

  // functions to add histograms to the registry
  void add(const HistogramSpec& histSpec);
  void add(char const* const name, char const* const title, const HistogramConfigSpec& histConfigSpec, bool callSumw2 = false);
  void add(char const* const name, char const* const title, HistType histType, std::vector<AxisSpec> axes, bool callSumw2 = false);
  void addClone(const std::string& source, const std::string& target);

  // function to query if name is already in use
  bool contains(char const* const name)
  {
    return contains(compile_time_hash(name), name);
  }

  // gets the underlying histogram pointer
  // we cannot automatically infer type here so it has to be explicitly specified
  // -> get<TH1>(), get<TH2>(), get<TH3>(), get<THn>(), get<THnSparse>(), get<TProfile>(), get<TProfile2D>(), get<TProfile3D>()
  /// @return the histogram registered with name @a name
  template <typename T>
  auto& get(char const* const name)
  {
    if (auto histPtr = std::get_if<std::shared_ptr<T>>(&mRegistryValue[getHistIndex(name)])) {
      return *histPtr;
    } else {
      throw runtime_error("Histogram type specified in get() does not match actual histogram type!");
    }
  }

  /// @return the histogram registered with name @a name
  template <typename T>
  auto& operator()(char const* const name)
  {
    return get<T>(name);
  }

  /// @return the associated OutputSpec
  OutputSpec const spec()
  {
    ConcreteDataMatcher matcher{"ATSK", "\0", 0};
    strncpy(matcher.description.str, mName.data(), 16);
    return OutputSpec{OutputLabel{mName}, matcher};
  }

  OutputRef ref()
  {
    return OutputRef{std::string{mName}, 0, o2::header::Stack{OutputObjHeader{mPolicy, OutputObjSourceType::HistogramRegistrySource, mTaskHash}}};
  }

  void setHash(uint32_t hash)
  {
    mTaskHash = hash;
  }

  TList* operator*();

  // fill hist with values
  template <typename... Ts>
  void fill(char const* const name, Ts&&... positionAndWeight)
  {
    std::visit([&positionAndWeight...](auto&& hist) { HistFiller::fillHistAny(hist, std::forward<Ts>(positionAndWeight)...); }, mRegistryValue[getHistIndex(name)]);
  }

  // fill hist with content of (filtered) table columns
  template <typename... Cs, typename T>
  void fill(char const* const name, const T& table, const o2::framework::expressions::Filter& filter)
  {
    std::visit([&table, &filter](auto&& hist) { HistFiller::fillHistAny<Cs...>(hist, table, filter); }, mRegistryValue[getHistIndex(name)]);
  }

  // get rough estimate for size of histogram stored in registry
  double getSize(char const* const name, double fillFraction = 1.)
  {
    double size{};
    std::visit([&fillFraction, &size](auto&& hist) { size = HistFiller::getSize(hist, fillFraction); }, mRegistryValue[getHistIndex(name)]);
    return size;
  }

  // get rough estimate for size of all histograms stored in registry
  double getSize(double fillFraction = 1.)
  {
    double size{};
    for (auto j = 0u; j < MAX_REGISTRY_SIZE; ++j) {
      std::visit([&fillFraction, &size](auto&& hist) { if(hist) { size += HistFiller::getSize(hist, fillFraction);} }, mRegistryValue[j]);
    }
    return size;
  }

  // print summary of the histograms stored in registry
  void print(bool showAxisDetails = false);

  // lookup distance counter for benchmarking
  mutable uint32_t lookup = 0;

 private:
  // create histogram from specification and insert it into the registry
  void insert(const HistogramSpec& histSpec)
  {
    const uint32_t i = imask(histSpec.id);
    for (auto j = 0u; j < MAX_REGISTRY_SIZE; ++j) {
      TObject* rawPtr = nullptr;
      std::visit([&](const auto& sharedPtr) { rawPtr = sharedPtr.get(); }, mRegistryValue[imask(j + i)]);
      if (!rawPtr) {
        registerName(histSpec.name);
        mRegistryKey[imask(j + i)] = histSpec.id;
        mRegistryValue[imask(j + i)] = HistFactory::createHistVariant(histSpec);
        lookup += j;
        return;
      }
    }
    LOGF(FATAL, "Internal array of HistogramRegistry %s is full.", mName);
  }

  // clone an existing histogram and insert it into the registry
  template <typename T>
  void insertClone(char const* const name, const std::shared_ptr<T>& originalHist)
  {
    const uint32_t id = compile_time_hash(name);
    const uint32_t i = imask(id);
    for (auto j = 0u; j < MAX_REGISTRY_SIZE; ++j) {
      TObject* rawPtr = nullptr;
      std::visit([&](const auto& sharedPtr) { rawPtr = sharedPtr.get(); }, mRegistryValue[imask(j + i)]);
      if (!rawPtr) {
        registerName(name);
        mRegistryKey[imask(j + i)] = id;
        mRegistryValue[imask(j + i)] = std::shared_ptr<T>(static_cast<T*>(originalHist->Clone(name)));
        lookup += j;
        return;
      }
    }
    LOGF(FATAL, "Internal array of HistogramRegistry %s is full.", mName);
  }

  constexpr uint32_t imask(uint32_t i) const
  {
    return i & MASK;
  }

  uint32_t getHistIndex(char const* const name)
  {
    const uint32_t id = compile_time_hash(name);
    const uint32_t i = imask(id);
    if (O2_BUILTIN_LIKELY(id == mRegistryKey[i])) {
      return i;
    }
    for (auto j = 1u; j < MAX_REGISTRY_SIZE; ++j) {
      if (id == mRegistryKey[imask(j + i)]) {
        return imask(j + i);
      }
    }
    throw runtime_error("No matching histogram found in HistogramRegistry!");
  }

  bool contains(const uint32_t id, char const* const name);

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

  // The maximum number of histograms in buffer is currently set to 512
  // which seems to be both reasonably large and allowing for very fast lookup
  static constexpr uint32_t MASK{0x1FF};
  static constexpr uint32_t MAX_REGISTRY_SIZE{MASK + 1};
  std::array<uint32_t, MAX_REGISTRY_SIZE> mRegistryKey{};
  std::array<HistPtr, MAX_REGISTRY_SIZE> mRegistryValue{};
};

} // namespace o2::framework
#endif // FRAMEWORK_HISTOGRAMREGISTRY_H_
