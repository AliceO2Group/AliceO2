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

#ifndef FRAMEWORK_HISTOGRAMSPEC_H_
#define FRAMEWORK_HISTOGRAMSPEC_H_

#include <string>
#include <utility>
#include <variant>
#include <optional>

#include "Framework/StringHelpers.h"
#include "Framework/Configurable.h"

#include "Framework/StepTHn.h"
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THn.h>
#include <THnSparse.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TProfile3D.h>

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
  kStepTHnF,
  kStepTHnD
};

// variant of all possible root pointers; here we use only the interface types since the underlying data representation (int,float,double,long,char) is irrelevant
using HistPtr = std::variant<std::shared_ptr<THn>, std::shared_ptr<THnSparse>, std::shared_ptr<TH3>, std::shared_ptr<TH2>, std::shared_ptr<TH1>, std::shared_ptr<TProfile3D>, std::shared_ptr<TProfile2D>, std::shared_ptr<TProfile>, std::shared_ptr<StepTHn>>;

//**************************************************************************************************
/**
 * Specification of an Axis.
 */
//**************************************************************************************************
// flag to mark variable bin size in configurable bin edges
constexpr double VARIABLE_WIDTH = 0.;

struct AxisSpec {
  AxisSpec(std::vector<double> binEdges_, std::optional<std::string> title_ = std::nullopt, std::optional<std::string> name_ = std::nullopt)
    : nBins(std::nullopt),
      binEdges(std::move(binEdges_)),
      title(std::move(title_)),
      name(std::move(name_))
  {
  }

  AxisSpec(int nBins_, double binMin_, double binMax_, std::optional<std::string> title_ = std::nullopt, std::optional<std::string> name_ = std::nullopt)
    : nBins(nBins_),
      binEdges({binMin_, binMax_}),
      title(std::move(title_)),
      name(std::move(name_))
  {
    if (binMin_ > binMax_) {
      LOG(fatal) << "Defined ill-defined axis";
    }
  }

  // first entry is assumed to be the number of bins; in case of variable size binning it must be set to zero
  AxisSpec(ConfigurableAxis binEdges_, std::optional<std::string> title_ = std::nullopt, std::optional<std::string> name_ = std::nullopt)
    : nBins(std::nullopt),
      binEdges(std::vector<double>(binEdges_)),
      title(std::move(title_)),
      name(std::move(name_))
  {
    if (binEdges.empty()) {
      return;
    }
    if (binEdges[0] != VARIABLE_WIDTH) {
      nBins = static_cast<int>(binEdges[0]);
      binEdges.resize(3); // nBins, lowerBound, upperBound, disregard whatever else is stored in vecotr
    }
    binEdges.erase(binEdges.begin()); // remove first entry that we assume to be number of bins
  }

  [[nodiscard]] long getNbins() const;

  /// Function to make the axis logarithmic
  void makeLogarithmic();

  /// Data members
  std::optional<int> nBins{};         /// Number of bins (only used for fixed bin width axis)
  std::vector<double> binEdges{};     /// Edges of the bin. For fixed bin width these are the limits of the binning
  std::optional<std::string> title{}; /// Optional title of the axis
  std::optional<std::string> name{};  /// Optional axis name for ndim histograms
};

//**************************************************************************************************
/**
 * Specification of a histogram configuration.
 */
//**************************************************************************************************
struct HistogramConfigSpec {
  HistogramConfigSpec(HistType type_, std::vector<AxisSpec> axes_, uint8_t nSteps_ = 1)
    : type(type_),
      axes(std::move(axes_)),
      nSteps(nSteps_)
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
    axes.emplace_back(nBins_, binMin_, binMax_, title_, name_);
  }

  void addAxis(std::vector<double> binEdges_, std::optional<std::string> title_ = std::nullopt, std::optional<std::string> name_ = std::nullopt)
  {
    axes.emplace_back(binEdges_, title_, name_);
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
  uint32_t nSteps{1}; // variable used only in StepTHn
};

//**************************************************************************************************
/**
 * Specification of a histogram.
 */
//**************************************************************************************************
struct HistogramSpec {
  HistogramSpec(char const* const name_, char const* const title_, HistogramConfigSpec config_, bool callSumw2_ = false)
    : name(name_),
      hash(compile_time_hash(name_)),
      title(title_),
      config(std::move(config_)),
      callSumw2(callSumw2_)
  {
  }

  HistogramSpec() = default;
  HistogramSpec(HistogramSpec const& other) = default;
  HistogramSpec(HistogramSpec&& other) = default;

  std::string name{};
  uint32_t hash = 0;
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
  static std::unique_ptr<T> createHist(const HistogramSpec& histSpec);

  // runtime version of the above
  static HistPtr createHistVariant(const HistogramSpec& histSpec);

  // helper function to get the axis via index for any type of root histogram
  template <typename T>
  static TAxis* getAxis(const int i, T* hist);
};

#define DECLAREEXT(HType) \
  extern template std::unique_ptr<HType> HistFactory::createHist<HType>(const HistogramSpec& histSpec);
DECLAREEXT(TH1D);
DECLAREEXT(TH1F);
DECLAREEXT(TH1I);
DECLAREEXT(TH1C);
DECLAREEXT(TH1S);
DECLAREEXT(TH2D);
DECLAREEXT(TH2F);
DECLAREEXT(TH2I);
DECLAREEXT(TH2C);
DECLAREEXT(TH2S);
DECLAREEXT(TH3D);
DECLAREEXT(TH3F);
DECLAREEXT(TH3I);
DECLAREEXT(TH3C);
DECLAREEXT(TH3S);
DECLAREEXT(THnD);
DECLAREEXT(THnF);
DECLAREEXT(THnI);
DECLAREEXT(THnC);
DECLAREEXT(THnS);
DECLAREEXT(THnL);
DECLAREEXT(THnSparseD);
DECLAREEXT(THnSparseF);
DECLAREEXT(THnSparseI);
DECLAREEXT(THnSparseC);
DECLAREEXT(THnSparseS);
DECLAREEXT(THnSparseL);
DECLAREEXT(TProfile);
DECLAREEXT(TProfile2D);
DECLAREEXT(TProfile3D);
DECLAREEXT(StepTHnF);
DECLAREEXT(StepTHnD)
#undef DECLAREEXT

} // namespace o2::framework
#endif // FRAMEWORK_HISTOGRAMSPEC_H_
