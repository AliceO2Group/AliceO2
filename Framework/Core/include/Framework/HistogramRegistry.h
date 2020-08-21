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
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"
#include "TFolder.h"

#include <string>
#include <variant>
namespace o2
{

namespace framework
{
// Most common histogram types
enum HistogramType : unsigned int {
  kTH1D = 0,
  kTH1F = 1,
  kTH1I = 2,
  kTH2D = 3,
  kTH2F = 4,
  kTH2I = 5,
  kTH3D = 6,
  kTH3F = 7,
  kTH3I = 8
};

/// Description of a single histogram axis
struct AxisSpec {
  AxisSpec(int nBins_, std::vector<double> bins_, std::string label_ = "")
    : nBins(nBins_),
      bins(bins_),
      binsEqual(false),
      label(label_)
  {
  }

  AxisSpec(int nBins_, double binMin_, double binMax_, std::string label_ = "")
    : nBins(nBins_),
      bins({binMin_, binMax_}),
      binsEqual(true),
      label(label_)
  {
  }

  AxisSpec() : nBins(1), binsEqual(false), bins(), label("") {}

  int nBins;
  std::vector<double> bins;
  bool binsEqual; // if true, then bins specify min and max for equidistant binning
  std::string label;
};

/// Data sctructure that will allow to construct a fully qualified TH* histogram
struct HistogramConfigSpec {
  HistogramConfigSpec(HistogramType type_, std::vector<AxisSpec> axes_)
    : type(type_),
      axes(axes_),
      binsEqual(axes.size() > 0 ? axes[0].binsEqual : false)
  {
  }

  HistogramConfigSpec()
    : type(HistogramType::kTH1F),
      axes(),
      binsEqual(false)
  {
  }
  HistogramConfigSpec(HistogramConfigSpec const& other) = default;
  HistogramConfigSpec(HistogramConfigSpec&& other) = default;

  HistogramType type;
  std::vector<AxisSpec> axes;
  bool binsEqual;
};

/// Data structure containing histogram specification for the HistogramRegistry
/// Contains hashed name as id for fast lookup
struct HistogramSpec {
  HistogramSpec(char const* const name_, char const* const readableName_, HistogramConfigSpec config_)
    : name(name_),
      readableName(readableName_),
      id(compile_time_hash(name_)),
      config(config_)
  {
  }

  HistogramSpec()
    : name(""),
      readableName(""),
      id(0),
      config()
  {
  }
  HistogramSpec(HistogramSpec const& other) = default;
  HistogramSpec(HistogramSpec&& other) = default;

  std::string name;
  std::string readableName;
  uint32_t id;
  HistogramConfigSpec config;
};

template <typename T>
std::unique_ptr<TH1> createTH1FromSpec(HistogramSpec const& spec)
{
  if (spec.config.axes.size() == 0) {
    throw runtime_error("No arguments available in spec to create a histogram");
  }
  if (spec.config.binsEqual) {
    return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins[0], spec.config.axes[0].bins[1]);
  }
  return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins.data());
}

template <typename T>
std::unique_ptr<TH2> createTH2FromSpec(HistogramSpec const& spec)
{
  if (spec.config.axes.size() == 0) {
    throw runtime_error("No arguments available in spec to create a histogram");
  }
  if (spec.config.binsEqual) {
    return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins[0], spec.config.axes[0].bins[1], spec.config.axes[1].nBins, spec.config.axes[1].bins[0], spec.config.axes[1].bins[1]);
  }
  return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins.data(), spec.config.axes[1].nBins, spec.config.axes[1].bins.data());
}

template <typename T>
std::unique_ptr<TH3> createTH3FromSpec(HistogramSpec const& spec)
{
  if (spec.config.axes.size() == 0) {
    throw runtime_error("No arguments available in spec to create a histogram");
  }
  if (spec.config.binsEqual) {
    return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins[0], spec.config.axes[0].bins[1], spec.config.axes[1].nBins, spec.config.axes[1].bins[0], spec.config.axes[1].bins[1], spec.config.axes[2].nBins, spec.config.axes[2].bins[0], spec.config.axes[2].bins[1]);
  }
  return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins.data(), spec.config.axes[1].nBins, spec.config.axes[1].bins.data(), spec.config.axes[2].nBins, spec.config.axes[2].bins.data());
}

/// Helper functions to fill histograms with expressions
template <typename C1, typename C2, typename C3, typename T>
void fill(TH1* hist, const T& table, const o2::framework::expressions::Filter& filter)
{
  auto filtered = o2::soa::Filtered<T>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)};
  for (auto& t : filtered) {
    hist->Fill(*(static_cast<C1>(t).getIterator()), *(static_cast<C2>(t).getIterator()), *(static_cast<C3>(t).getIterator()));
  }
}

template <typename C1, typename C2, typename T>
void fill(TH1* hist, const T& table, const o2::framework::expressions::Filter& filter)
{
  auto filtered = o2::soa::Filtered<T>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)};
  for (auto& t : filtered) {
    hist->Fill(*(static_cast<C1>(t).getIterator()), *(static_cast<C2>(t).getIterator()));
  }
}

template <typename C, typename T>
void fill(TH1* hist, const T& table, const o2::framework::expressions::Filter& filter)
{
  auto filtered = o2::soa::Filtered<T>{{table.asArrowTable()}, o2::framework::expressions::createSelection(table.asArrowTable(), filter)};
  for (auto& t : filtered) {
    hist->Fill(*(static_cast<C>(t).getIterator()));
  }
}

using HistogramCreationCallback = std::function<std::unique_ptr<TH1>(HistogramSpec const& spec)>;

// Wrapper to avoid multiple function definitinions error
struct HistogramCallbacks {
  static HistogramCreationCallback createTH1D()
  {
    return [](HistogramSpec const& spec) -> std::unique_ptr<TH1> {
      return createTH1FromSpec<TH1D>(spec);
    };
  }

  static HistogramCreationCallback createTH1F()
  {
    return [](HistogramSpec const& spec) -> std::unique_ptr<TH1> {
      return createTH1FromSpec<TH1F>(spec);
    };
  }

  static HistogramCreationCallback createTH1I()
  {
    return [](HistogramSpec const& spec) -> std::unique_ptr<TH1> {
      return createTH1FromSpec<TH1I>(spec);
    };
  }

  static HistogramCreationCallback createTH2D()
  {
    return [](HistogramSpec const& spec) -> std::unique_ptr<TH1> {
      return createTH2FromSpec<TH2D>(spec);
    };
  }

  static HistogramCreationCallback createTH2F()
  {
    return [](HistogramSpec const& spec) -> std::unique_ptr<TH1> {
      return createTH2FromSpec<TH2F>(spec);
    };
  }

  static HistogramCreationCallback createTH2I()
  {
    return [](HistogramSpec const& spec) -> std::unique_ptr<TH1> {
      return createTH2FromSpec<TH2I>(spec);
    };
  }

  static HistogramCreationCallback createTH3D()
  {
    return [](HistogramSpec const& spec) -> std::unique_ptr<TH1> {
      return createTH3FromSpec<TH3D>(spec);
    };
  }

  static HistogramCreationCallback createTH3F()
  {
    return [](HistogramSpec const& spec) -> std::unique_ptr<TH1> {
      return createTH3FromSpec<TH3F>(spec);
    };
  }

  static HistogramCreationCallback createTH3I()
  {
    return [](HistogramSpec const& spec) -> std::unique_ptr<TH1> {
      return createTH3FromSpec<TH3I>(spec);
    };
  }
};

/// Histogram registry for an analysis task that allows to define needed histograms
/// and serves as the container/wrapper to fill them
class HistogramRegistry
{
 public:
  HistogramRegistry(char const* const name_, bool enable, std::vector<HistogramSpec> specs, OutputObjHandlingPolicy policy_ = OutputObjHandlingPolicy::AnalysisObject)
    : name(name_),
      policy(policy_),
      enabled(enable),
      mRegistryKey(),
      mRegistryValue(),
      mHistogramCreationCallbacks({HistogramCallbacks::createTH1D(),
                                   HistogramCallbacks::createTH1F(),
                                   HistogramCallbacks::createTH1I(),
                                   HistogramCallbacks::createTH2D(),
                                   HistogramCallbacks::createTH2F(),
                                   HistogramCallbacks::createTH2I(),
                                   HistogramCallbacks::createTH3D(),
                                   HistogramCallbacks::createTH3F(),
                                   HistogramCallbacks::createTH3I()})
  {
    mRegistryKey.fill(0u);
    for (auto& spec : specs) {
      insert(spec);
    }
  }

  /// @return the histogram registered with name @a name
  auto& get(char const* const name) const
  {
    const uint32_t id = compile_time_hash(name);
    const uint32_t i = imask(id);
    if (O2_BUILTIN_LIKELY(id == mRegistryKey[i])) {
      return mRegistryValue[i];
    }
    for (auto j = 1u; j < MAX_REGISTRY_SIZE; ++j) {
      if (id == mRegistryKey[imask(j + i)]) {
        return mRegistryValue[imask(j + i)];
      }
    }
    throw runtime_error("No match found!");
  }

  /// @return the histogram registered with name @a name
  auto& operator()(char const* const name) const
  {
    return get(name);
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

  TFolder* operator*()
  {
    TFolder* folder = new TFolder(this->name.c_str(), this->name.c_str());
    for (auto j = 0u; j < MAX_REGISTRY_SIZE; ++j) {
      if (mRegistryValue[j].get() != nullptr) {
        auto hist = mRegistryValue[j].get();
        folder->Add(hist);
      }
    }
    folder->SetOwner();
    return folder;
  }

  /// fill the histogram with an expression
  template <typename C1, typename C2, typename C3, typename T>
  void fill(char const* const name, const T& table, const o2::framework::expressions::Filter& filter)
  {
    TH1* hist = get(name).get();
    framework::fill<C1, C2, C3>(hist, table, filter);
  }

  template <typename C1, typename C2, typename T>
  void fill(char const* const name, const T& table, const o2::framework::expressions::Filter& filter)
  {
    TH1* hist = get(name).get();
    framework::fill<C1, C2>(hist, table, filter);
  }

  template <typename C, typename T>
  void fill(char const* const name, const T& table, const o2::framework::expressions::Filter& filter)
  {
    TH1* hist = get(name).get();
    framework::fill<C>(hist, table, filter);
  }

  /// lookup distance counter for benchmarking
  mutable uint32_t lookup = 0;

 private:
  void insert(HistogramSpec& spec)
  {
    uint32_t i = imask(spec.id);
    for (auto j = 0u; j < MAX_REGISTRY_SIZE; ++j) {
      if (mRegistryValue[imask(j + i)].get() == nullptr) {
        mRegistryKey[imask(j + i)] = spec.id;
        mRegistryValue[imask(j + i)] = mHistogramCreationCallbacks[spec.config.type](spec);
        lookup += j;
        return;
      }
    }
    throw runtime_error("Internal array is full.");
  }

  inline constexpr uint32_t imask(uint32_t i) const
  {
    return i & mask;
  }
  std::string name;
  bool enabled;
  OutputObjHandlingPolicy policy;
  uint32_t taskHash;

  /// The maximum number of histograms in buffer is currently set to 512
  /// which seems to be both reasonably large and allowing for very fast lookup
  static constexpr uint32_t mask = 0x1FF;
  static constexpr uint32_t MAX_REGISTRY_SIZE = mask + 1;
  std::array<uint32_t, MAX_REGISTRY_SIZE> mRegistryKey;
  std::array<std::unique_ptr<TH1>, MAX_REGISTRY_SIZE> mRegistryValue;
  std::vector<HistogramCreationCallback> mHistogramCreationCallbacks;
};

} // namespace framework

} // namespace o2

#endif // FRAMEWORK_HISTOGRAMREGISTRY_H_
