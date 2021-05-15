// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_HISTOGRAMSPEC_H_
#define FRAMEWORK_HISTOGRAMSPEC_H_

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

#include "Framework/StepTHn.h"
#include "Framework/Configurable.h"
#include "Framework/StringHelpers.h"
#include "Framework/RuntimeError.h"

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
// Flag to mark variable bin size in configurable bin edges
constexpr int VARIABLE_WIDTH = 0;

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

  // first entry is assumed to be the number of bins; in case of variable size binning it must be set to zero
  AxisSpec(ConfigurableAxis binEdges_, std::optional<std::string> title_ = std::nullopt, std::optional<std::string> name_ = std::nullopt)
    : nBins(std::nullopt),
      binEdges(std::vector<double>(binEdges_)),
      title(title_),
      name(name_)
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
  HistogramConfigSpec(HistType type_, std::vector<AxisSpec> axes_, uint8_t nSteps_ = 1)
    : type(type_),
      axes(axes_),
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
      config(config_),
      callSumw2(callSumw2_)
  {
  }

  HistogramSpec()
    : name(""),
      hash(0),
      config()
  {
  }
  HistogramSpec(HistogramSpec const& other) = default;
  HistogramSpec(HistogramSpec&& other) = default;

  std::string name{};
  uint32_t hash{};
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
  static std::unique_ptr<T> createHist(const HistogramSpec& histSpec)
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
    std::unique_ptr<T> hist{generateHist<T>(histSpec.name, histSpec.title, nAxes, nBins, lowerBounds, upperBounds, histSpec.config.nSteps)};
    if (!hist) {
      LOGF(FATAL, "The number of dimensions specified for histogram %s does not match the type.", histSpec.name);
      return nullptr;
    }

    // set axis properties
    for (std::size_t i = 0; i < nAxes; i++) {
      TAxis* axis{getAxis(i, hist.get())};
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
    if (auto hist = castToVariant(std::shared_ptr<T>(std::move(createHist<T>(histSpec))))) {
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
  static TAxis* getAxis(const int i, T* hist)
  {
    if constexpr (std::is_base_of_v<THnBase, T> || std::is_base_of_v<StepTHn, T>) {
      return hist->GetAxis(i);
    } else {
      if (i == 0) {
        return hist->GetXaxis();
      } else if (i == 1) {
        return hist->GetYaxis();
      } else if (i == 2) {
        return hist->GetZaxis();
      } else {
        return nullptr;
      }
    }
  }

 private:
  static const std::map<HistType, std::function<HistPtr(const HistogramSpec&)>> HistogramCreationCallbacks;

  // helper function to generate the actual histograms
  template <typename T>
  static T* generateHist(const std::string& name, const std::string& title, const std::size_t nDim,
                         const int nBins[], const double lowerBounds[], const double upperBounds[], const int nSteps = 1)
  {
    if constexpr (std::is_base_of_v<StepTHn, T>) {
      return new T(name.data(), title.data(), nSteps, nDim, nBins, lowerBounds, upperBounds);
    } else if constexpr (std::is_base_of_v<THnBase, T>) {
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
      return castToVariant<THn, THnSparse, TProfile3D, TH3, TProfile2D, TH2, TProfile, TH1, StepTHn>(obj);
    }
    return std::nullopt;
  }
};

} // namespace o2::framework
#endif // FRAMEWORK_HISTOGRAMSPEC_H_
