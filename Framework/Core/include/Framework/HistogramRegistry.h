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
#include "Framework/Logger.h"
#include "Framework/FunctionalHelpers.h"
#include "Framework/StringHelpers.h"

#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "THn.h"
#include "THnSparse.h"

#include <string>

namespace o2
{

namespace framework
{

struct HistogramConfigSpec {
  HistogramConfigSpec(char const* const kind_, int nBins_, double xmin_, double xmax_)
    : kind(kind_),
      nBins(nBins_),
      xmin(xmin_),
      xmax(xmax_)
  {
  }
  template <typename... T>
  HistogramConfigSpec(T... args)
  {
  }
  std::string kind;
  int nBins;
  double xmin;
  double xmax;
};

struct HistogramSpec {
  HistogramSpec(char const* const name_, char const* const readableName_, HistogramConfigSpec config_)
    : name(name_),
      readableName(readableName_),
      id(compile_time_hash(name_)),
      config(config_)
  {
  }
  std::string name;
  std::string readableName;
  uint32_t id;
  HistogramConfigSpec config;
};

class HistogramRegistry
{
 public:
  HistogramRegistry(char const* name_, bool enable, std::vector<HistogramSpec> specs)
    : name(name_),
      enabled(enable)
  {
    for (auto spec : specs) {
      //            LOGP(ERROR,"{}",compile_time_hash(spec.name.data()));
      //            LOGP(ERROR,"{}",spec.id);
      histograms.insert(std::make_pair(spec.id,
                                       std::make_unique<TH1F>(spec.name.data(), spec.readableName.data(), spec.config.nBins, spec.config.xmin, spec.config.xmax)));
    }
  }
  auto& get(const char* const name_)
  {
    //        LOGP(ERROR,"{}",compile_time_hash(name_));
    return histograms[compile_time_hash(name_)];
  }

 private:
  std::string name;
  bool enabled;
  std::unordered_map<uint32_t, std::unique_ptr<TH1>> histograms;
};

} // namespace framework

} // namespace o2

#endif // FRAMEWORK_HISTOGRAMREGISTRY_H_
