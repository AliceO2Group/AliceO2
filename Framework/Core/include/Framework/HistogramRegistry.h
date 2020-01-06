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
/// Data sctructure that will allow to construct a fully qualified TH* histogram
/// Currently only supports TH1F
struct HistogramConfigSpec {
  HistogramConfigSpec(char const* const kind_, unsigned int nBins_, double xmin_, double xmax_)
    : kind(kind_),
      nBins(nBins_),
      xmin(xmin_),
      xmax(xmax_)
  {
  }

  HistogramConfigSpec()
    : kind(""),
      nBins(1),
      xmin(0),
      xmax(1)
  {
  }
  HistogramConfigSpec(HistogramConfigSpec const& other) = default;
  HistogramConfigSpec(HistogramConfigSpec&& other) = default;

  std::string kind;
  unsigned int nBins;
  double xmin;
  double xmax;
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
        mRegistryValue[imask(j + i)] = {std::make_unique<TH1F>(spec.name.data(), spec.readableName.data(), spec.config.nBins, spec.config.xmin, spec.config.xmax)};
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
