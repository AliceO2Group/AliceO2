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
#include "Framework/OutputSpec.h"
#include "Framework/StringHelpers.h"
#include "Framework/TableBuilder.h"

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include <string>
#include <variant>
namespace o2
{

namespace framework
{
// Most common histogram types
enum HistogramType {
  kTH1D,
  kTH1F,
  kTH1I,
  kTH2D,
  kTH2F,
  kTH2I,
  kTH3D,
  kTH3F,
  kTH3I
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

class HistogramFactory
{
 public:
  static std::unique_ptr<TH1> create(HistogramSpec& spec)
  {
    const auto& it = lookup().find(spec.config.type);
    if (it == lookup().end()) {
      return nullptr;
    }
    return std::move(it->second->createImpl(spec));
  }

 protected:
  static std::map<HistogramType, HistogramFactory*>& lookup()
  {
    static std::map<HistogramType, HistogramFactory*> histMap;
    return histMap;
  }

 private:
  virtual std::unique_ptr<TH1> createImpl(HistogramSpec const& spec) = 0;
};

template <typename T>
class HistogramFactoryImpl : public HistogramFactory
{
 public:
  HistogramFactoryImpl(HistogramType type)
    : position(this->lookup().insert(std::make_pair(type, this)).first)
  {
  }

  ~HistogramFactoryImpl()
  {
    this->lookup().erase(position);
  }

 private:
  std::unique_ptr<TH1> createImpl(HistogramSpec const& spec) override
  {
    if (spec.config.axes.size() == 0) {
      throw std::runtime_error("No arguments available in spec to create a histogram");
    }
    if constexpr (std::is_base_of_v<TH3, T>) {
      if (spec.config.binsEqual) {
        return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins[0], spec.config.axes[0].bins[1], spec.config.axes[1].nBins, spec.config.axes[1].bins[0], spec.config.axes[1].bins[1], spec.config.axes[2].nBins, spec.config.axes[2].bins[0], spec.config.axes[2].bins[1]);
      }
      return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins.data(), spec.config.axes[1].nBins, spec.config.axes[1].bins.data(), spec.config.axes[2].nBins, spec.config.axes[2].bins.data());
    } else if constexpr (std::is_base_of_v<TH2, T>) {
      if (spec.config.binsEqual) {
        return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins[0], spec.config.axes[0].bins[1], spec.config.axes[1].nBins, spec.config.axes[1].bins[0], spec.config.axes[1].bins[1]);
      }
      return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins.data(), spec.config.axes[1].nBins, spec.config.axes[1].bins.data());
    } else if constexpr (std::is_base_of_v<TH1, T>) {
      if (spec.config.binsEqual) {
        return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins[0], spec.config.axes[0].bins[1]);
      }
      return std::make_unique<T>(spec.name.data(), spec.readableName.data(), spec.config.axes[0].nBins, spec.config.axes[0].bins.data());
    }
  }

  typename std::map<HistogramType, HistogramFactory*>::iterator position;
};

HistogramFactoryImpl<TH1D> const hf1d(HistogramType::kTH1D);
HistogramFactoryImpl<TH1F> const hf1f(HistogramType::kTH1F);
HistogramFactoryImpl<TH1I> const hf1i(HistogramType::kTH1I);
HistogramFactoryImpl<TH2D> const hf2d(HistogramType::kTH2D);
HistogramFactoryImpl<TH2F> const hf2f(HistogramType::kTH2F);
HistogramFactoryImpl<TH2I> const hf2i(HistogramType::kTH2I);
HistogramFactoryImpl<TH3D> const hf3d(HistogramType::kTH3D);
HistogramFactoryImpl<TH3F> const hf3f(HistogramType::kTH3F);
HistogramFactoryImpl<TH3I> const hf3i(HistogramType::kTH3I);

/// Histogram registry for an analysis task that allows to define needed histograms
/// and serves as the container/wrapper to fill them
class HistogramRegistry
{
 public:
  HistogramRegistry(char const* const name_, bool enable, std::vector<HistogramSpec> specs)
    : name(name_),
      enabled(enable),
      mRegistryKey(),
      mRegistryValue()
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
    throw std::runtime_error("No match found!");
  }

  /// @return the histogram registered with name @a name
  auto& operator()(char const* const name) const
  {
    return get(name);
  }

  // @return the associated OutputSpec
  OutputSpec const spec()
  {
    ConcreteDataMatcher matcher{"HIST", "\0", 0};
    strncpy(matcher.description.str, this->name.data(), 16);
    return OutputSpec{OutputLabel{this->name}, matcher};
  }

  OutputRef ref()
  {
    return OutputRef{this->name, 0};
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
        mRegistryValue[imask(j + i)] = HistogramFactory::create(spec);
        lookup += j;
        return;
      }
    }
    throw std::runtime_error("Internal array is full.");
  }

  inline constexpr uint32_t imask(uint32_t i) const
  {
    return i & mask;
  }
  std::string name;
  bool enabled;

  /// The maximum number of histograms in buffer is currently set to 512
  /// which seems to be both reasonably large and allowing for very fast lookup
  static constexpr uint32_t mask = 0x1FF;
  static constexpr uint32_t MAX_REGISTRY_SIZE = mask + 1;
  std::array<uint32_t, MAX_REGISTRY_SIZE> mRegistryKey;
  std::array<std::unique_ptr<TH1>, MAX_REGISTRY_SIZE> mRegistryValue;
};

} // namespace framework

} // namespace o2

#endif // FRAMEWORK_HISTOGRAMREGISTRY_H_
