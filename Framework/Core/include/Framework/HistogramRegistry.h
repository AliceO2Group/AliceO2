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

#include "TClass.h"

#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THn.h>
#include <THnSparse.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TProfile3D.h>

#include <TFolder.h>
//#include <TObjArray.h>
//#include <TList.h>

#include <string>
#include <variant>

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
  AxisSpec(std::vector<double> binEdges_, std::string title_ = "", std::optional<std::string> name_ = std::nullopt)
    : nBins(std::nullopt),
      binEdges(binEdges_),
      title(title_),
      name(name_)
  {
  }

  AxisSpec(int nBins_, double binMin_, double binMax_, std::string title_ = "", std::optional<std::string> name_ = std::nullopt)
    : nBins(nBins_),
      binEdges({binMin_, binMax_}),
      title(title_),
      name(name_)
  {
  }

  std::optional<int> nBins{};
  std::vector<double> binEdges{};
  std::string title{};
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

  void addAxis(int nBins_, const double binMin_, const double binMax_, const std::string& title_ = "", const std::optional<std::string>& name_ = std::nullopt)
  {
    axes.push_back({nBins_, binMin_, binMax_, title_, name_});
  }

  void addAxis(const std::vector<double>& binEdges_, const std::string& title_, const std::string& name_)
  {
    axes.push_back({binEdges_, title_, name_});
  }

  void addAxes(const std::vector<AxisSpec>& axes_)
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

  // create histogram of type T with the axes defined in HistogramConfigSpec
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
      LOGF(FATAL, "The number of specified dimensions does not match the type.");
      return nullptr;
    }

    // set axis properties
    for (std::size_t i = 0; i < nAxes; i++) {
      TAxis* axis{getAxis(i, hist)};
      if (axis) {
        axis->SetTitle(histSpec.config.axes[i].title.data());

        // this helps to have axes not only called 0,1,2... in ndim histos
        if constexpr (std::is_base_of_v<THnBase, T>) {
          if (histSpec.config.axes[i].name)
            axis->SetName((std::string(axis->GetName()) + "-" + *histSpec.config.axes[i].name).data());
        }

        // move the bin edges in case a variable binning was requested
        if (!histSpec.config.axes[i].nBins) {
          if (!std::is_sorted(std::begin(histSpec.config.axes[i].binEdges), std::end(histSpec.config.axes[i].binEdges))) {
            LOGF(FATAL, "The bin edges specified for axis %s in histogram %s are not in increasing order!", (histSpec.config.axes[i].name) ? *histSpec.config.axes[i].name : histSpec.config.axes[i].title, histSpec.name);
            return nullptr;
          }
          axis->Set(nBins[i], histSpec.config.axes[i].binEdges.data());
        }
      }
    }
    if (histSpec.callSumw2)
      hist->Sumw2();

    return hist;
  }

  // create histogram and return it casted to the correct alternative held in HistPtr variant
  template <typename T>
  static HistPtr createHistVariant(HistogramSpec const& histSpec)
  {
    if (auto hist = castToVariant(createHist<T>(histSpec)))
      return *hist;
    else
      throw runtime_error("Histogram was not created properly.");
  }

  // runtime version of the above
  static HistPtr createHistVariant(HistogramSpec const& histSpec)
  {
    if (histSpec.config.type == HistType::kUndefinedHist)
      throw runtime_error("Histogram type was not specified.");
    else
      return HistogramCreationCallbacks.at(histSpec.config.type)(histSpec);
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

  // helper function to get the axis via index for any type of root histogram
  template <typename T>
  static TAxis* getAxis(const int i, std::shared_ptr<T>& hist)
  {
    if constexpr (std::is_base_of_v<THnBase, T>) {
      return hist->GetAxis(i);
    } else {
      return (i == 0) ? hist->GetXaxis()
                      : (i == 1) ? hist->GetYaxis() : (i == 2) ? hist->GetZaxis() : nullptr;
    }
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
  template <bool useWeight = false, typename T, typename... Ts>
  static void fillHistAny(std::shared_ptr<T>& hist, const Ts&... positionAndWeight)
  {
    constexpr int nDim = sizeof...(Ts) - useWeight;

    constexpr bool validTH3 = (std::is_same_v<TH3, T> && nDim == 3);
    constexpr bool validTH2 = (std::is_same_v<TH2, T> && nDim == 2);
    constexpr bool validTH1 = (std::is_same_v<TH1, T> && nDim == 1);
    constexpr bool validTProfile3D = (std::is_same_v<TProfile3D, T> && nDim == 4);
    constexpr bool validTProfile2D = (std::is_same_v<TProfile2D, T> && nDim == 3);
    constexpr bool validTProfile = (std::is_same_v<TProfile, T> && nDim == 2);

    constexpr bool validSimpleFill = validTH1 || validTH2 || validTH3 || validTProfile || validTProfile2D || validTProfile3D;
    // unfortunately we dont know at compile the dimension of THn(Sparse)
    constexpr bool validComplexFill = std::is_base_of_v<THnBase, T>;

    if constexpr (validSimpleFill) {
      hist->Fill(static_cast<double>(positionAndWeight)...);
    } else if constexpr (validComplexFill) {
      // savety check for n dimensional histograms (runtime overhead)
      if (hist->GetNdimensions() != nDim)
        throw runtime_error("The number of position (and weight) arguments does not match the histogram dimensions!");

      double tempArray[sizeof...(Ts)] = {static_cast<double>(positionAndWeight)...};
      if constexpr (useWeight)
        hist->Fill(tempArray, tempArray[sizeof...(Ts) - 1]);
      else
        hist->Fill(tempArray);
    } else {
      throw runtime_error("The number of position (and weight) arguments does not match the histogram dimensions!");
    }
  }

  // fill any type of histogram with columns (Cs) of a filtered table (if weight is requested it must reside the last specified column)
  template <bool useWeight = false, typename... Cs, typename R, typename T>
  static void fillHistAnyTable(std::shared_ptr<R>& hist, const T& table, const o2::framework::expressions::Filter& filter)
  {
    auto filtered = o2::soa::Filtered<T>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)};
    for (auto& t : filtered) {
      fillHistAny<useWeight>(hist, (*(static_cast<Cs>(t).getIterator()))...);
    }
  }
};

//**************************************************************************************************
/**
 * Histogram registry that can be used to store and fill histograms of any type.
 */
//**************************************************************************************************
class HistogramRegistry
{
 public:
  HistogramRegistry(char const* const name_, bool enable, std::vector<HistogramSpec> histSpecs_, OutputObjHandlingPolicy policy_ = OutputObjHandlingPolicy::AnalysisObject) : name(name_),
                                                                                                                                                                              policy(policy_),
                                                                                                                                                                              enabled(enable),
                                                                                                                                                                              mRegistryKey(),
                                                                                                                                                                              mRegistryValue()
  {
    mRegistryKey.fill(0u);
    for (auto& histSpec : histSpecs_) {
      insert(histSpec);
    }
  }

  void add(HistogramSpec&& histSpec_)
  {
    insert(histSpec_);
  }

  // gets the underlying histogram pointer
  // we cannot automatically infer type here so it has to be explicitly specified
  // -> get<TH1>(), get<TH2>(), get<TH3>(), get<THn>(), get<THnSparse>(), get<TProfile>(), get<TProfile2D>(), get<TProfile3D>()
  /// @return the histogram registered with name @a name
  template <typename T>
  auto& get(char const* const name)
  {
    const uint32_t id = compile_time_hash(name);
    const uint32_t i = imask(id);
    if (O2_BUILTIN_LIKELY(id == mRegistryKey[i])) {
      return *std::get_if<std::shared_ptr<T>>(&mRegistryValue[i]);
    }
    for (auto j = 1u; j < MAX_REGISTRY_SIZE; ++j) {
      if (id == mRegistryKey[imask(j + i)]) {
        return *std::get_if<std::shared_ptr<T>>(&mRegistryValue[imask(j + i)]);
      }
    }
    throw runtime_error("No matching histogram found in HistogramRegistry!");
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
    ConcreteDataMatcher matcher{"HIST", "\0", 0};
    strncpy(matcher.description.str, this->name.data(), 16);
    return OutputSpec{OutputLabel{this->name}, matcher};
  }

  OutputRef ref()
  {
    return OutputRef{std::string{this->name}, 0, o2::header::Stack{OutputObjHeader{policy, taskHash}}};
  }
  void setHash(uint32_t hash)
  {
    taskHash = hash;
  }

  // TODO: maybe support also TDirectory,TList,TObjArray?, find a way to write to file in 'single key' mode (arg in WriteObjAny)
  TFolder* operator*()
  {
    TFolder* folder = new TFolder(this->name.c_str(), this->name.c_str());
    for (auto j = 0u; j < MAX_REGISTRY_SIZE; ++j) {
      TObject* rawPtr = nullptr;
      std::visit([&](const auto& sharedPtr) { rawPtr = sharedPtr.get(); }, mRegistryValue[j]);
      if (rawPtr) {
        folder->Add(rawPtr);
      }
    }
    folder->SetOwner(false); // object deletion will be handled by shared_ptrs
    return folder;
  }

  // fill hist with values
  template <bool useWeight = false, typename... Ts>
  void fill(char const* const name, Ts&&... positionAndWeight)
  {
    const uint32_t id = compile_time_hash(name);
    const uint32_t i = imask(id);
    if (O2_BUILTIN_LIKELY(id == mRegistryKey[i])) {
      std::visit([this, &positionAndWeight...](auto&& hist) { HistFiller::fillHistAny<useWeight>(hist, std::forward<Ts>(positionAndWeight)...); }, mRegistryValue[i]);
      return;
    }
    for (auto j = 1u; j < MAX_REGISTRY_SIZE; ++j) {
      if (id == mRegistryKey[imask(j + i)]) {
        std::visit([this, &positionAndWeight...](auto&& hist) { HistFiller::fillHistAny<useWeight>(hist, std::forward<Ts>(positionAndWeight)...); }, mRegistryValue[imask(j + i)]);
        return;
      }
    }
    throw runtime_error("No matching histogram found in HistogramRegistry!");
  }

  template <typename... Ts>
  void fillWeight(char const* const name, Ts&&... positionAndWeight)
  {
    fill<true>(name, std::forward<Ts>(positionAndWeight)...);
  }

  // fill hist with content of (filtered) table columns
  template <typename... Cs, typename T>
  void fill(char const* const name, const T& table, const o2::framework::expressions::Filter& filter)
  {
    fillTable<false, Cs...>(name, table, filter);
  }
  template <typename... Cs, typename T>
  void fillWeight(char const* const name, const T& table, const o2::framework::expressions::Filter& filter)
  {
    fillTable<true, Cs...>(name, table, filter);
  }

  // this is for internal use only because we dont want user to have to specify 'useWeight' argument
  template <bool useWeight = false, typename... Cs, typename T>
  void fillTable(char const* const name, const T& table, const o2::framework::expressions::Filter& filter)
  {
    const uint32_t id = compile_time_hash(name);
    const uint32_t i = imask(id);
    if (O2_BUILTIN_LIKELY(id == mRegistryKey[i])) {
      std::visit([this, &table, &filter](auto&& hist) { HistFiller::fillHistAnyTable<useWeight, Cs...>(hist, table, filter); }, mRegistryValue[i]);
      return;
    }
    for (auto j = 1u; j < MAX_REGISTRY_SIZE; ++j) {
      if (id == mRegistryKey[imask(j + i)]) {
        std::visit([this, &table, &filter](auto&& hist) { HistFiller::fillHistAnyTable<useWeight, Cs...>(hist, table, filter); }, mRegistryValue[imask(j + i)]);
        return;
      }
    }
    throw runtime_error("No matching histogram found in HistogramRegistry!");
  }

  /// lookup distance counter for benchmarking
  mutable uint32_t lookup = 0;

 private:
  void insert(HistogramSpec& histSpec)
  {
    uint32_t i = imask(histSpec.id);
    for (auto j = 0u; j < MAX_REGISTRY_SIZE; ++j) {
      TObject* rawPtr = nullptr;
      std::visit([&](const auto& sharedPtr) { rawPtr = sharedPtr.get(); }, mRegistryValue[imask(j + i)]);
      if (!rawPtr) {
        mRegistryKey[imask(j + i)] = histSpec.id;
        mRegistryValue[imask(j + i)] = HistFactory::createHistVariant(histSpec);
        lookup += j;
        return;
      }
    }
    throw runtime_error("Internal array of HistogramRegistry is full.");
  }

  inline constexpr uint32_t imask(uint32_t i) const
  {
    return i & mask;
  }

  std::string name{};
  bool enabled{};
  OutputObjHandlingPolicy policy{};
  uint32_t taskHash{};

  /// The maximum number of histograms in buffer is currently set to 512
  /// which seems to be both reasonably large and allowing for very fast lookup
  static constexpr uint32_t mask = 0x1FF;
  static constexpr uint32_t MAX_REGISTRY_SIZE = mask + 1;
  std::array<uint32_t, MAX_REGISTRY_SIZE> mRegistryKey{};
  std::array<HistPtr, MAX_REGISTRY_SIZE> mRegistryValue{};
};

} // namespace o2::framework
#endif // FRAMEWORK_HISTOGRAMREGISTRY_H_
