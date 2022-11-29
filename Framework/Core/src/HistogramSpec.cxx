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

#include "Framework/HistogramSpec.h"
#include "Framework/RuntimeError.h"

namespace o2::framework
{

void AxisSpec::makeLogarithmic()
{
  if (binEdges.size() > 2) {
    LOG(fatal) << "Cannot make a variabled bin width axis logaritmic";
  }

  const double min = binEdges[0];
  if (min <= 0.) {
    LOG(fatal) << "Cannot have the first bin limit of the log. axis below 0: " << min;
  }
  const double max = binEdges[1];
  binEdges.clear();
  const double logmin = std::log10(min);
  const double logmax = std::log10(max);
  const int nbins = nBins.value();
  const double logdelta = (logmax - logmin) / (static_cast<double>(nbins));
  LOG(debug) << "Making a logaritmic binning from " << min << " to " << max << " with " << nbins << " bins";
  for (int i = 0; i < nbins + 1; i++) {
    const auto nextEdge = std::pow(10, logmin + i * logdelta);
    LOG(debug) << i << "/" << nbins - 1 << ": " << nextEdge;
    binEdges.push_back(nextEdge);
  }
  nBins = std::nullopt;
}

long AxisSpec::getNbins() const
{
  // return the number of bins
  if (nBins.has_value()) {
    return *nBins;
  }
  return binEdges.size() - 1;
}

// helper function to generate the actual histograms
template <typename T>
T* generateHist(const std::string& name, const std::string& title, const std::size_t nDim,
                const int nBins[], const double lowerBounds[], const double upperBounds[], const int nSteps)
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

// main function for creating arbitrary histograms
template <typename T, typename BASE>
static std::unique_ptr<BASE> createHistFull(const HistogramSpec& histSpec)
{
  constexpr std::size_t MAX_DIM{10};
  const std::size_t nAxes{histSpec.config.axes.size()};
  if (nAxes == 0 || nAxes > MAX_DIM) {
    LOGF(fatal, "The histogram specification contains no (or too many) axes.");
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
  std::unique_ptr<BASE> hist{generateHist<T>(histSpec.name, histSpec.title, nAxes, nBins, lowerBounds, upperBounds, histSpec.config.nSteps)};
  if (!hist) {
    LOGF(fatal, "The number of dimensions specified for histogram %s does not match the type.", histSpec.name);
  }

  // set axis properties
  for (std::size_t i = 0; i < nAxes; i++) {
    TAxis* axis{HistFactory::getAxis(i, hist.get())};
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
          LOGF(fatal, "The bin edges in histogram %s are not in increasing order!", histSpec.name);
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

template <typename T>
std::unique_ptr<T> HistFactory::createHist(const HistogramSpec& histSpec)
{
  return createHistFull<T, T>(histSpec);
}

// define histogram callbacks for runtime histogram creation
#define CREATE_HIST_CASE(HType, BType) \
  case k##HType:                       \
    return std::shared_ptr<BType>(createHistFull<HType, BType>(histSpec));

// runtime version of the above
HistPtr HistFactory::createHistVariant(const HistogramSpec& histSpec)
{
  switch (histSpec.config.type) {
    CREATE_HIST_CASE(TH1D, TH1)
    CREATE_HIST_CASE(TH1F, TH1)
    CREATE_HIST_CASE(TH1I, TH1)
    CREATE_HIST_CASE(TH1C, TH1)
    CREATE_HIST_CASE(TH1S, TH1)

    CREATE_HIST_CASE(TH2D, TH2)
    CREATE_HIST_CASE(TH2F, TH2)
    CREATE_HIST_CASE(TH2I, TH2)
    CREATE_HIST_CASE(TH2C, TH2)
    CREATE_HIST_CASE(TH2S, TH2)

    CREATE_HIST_CASE(TH3D, TH3)
    CREATE_HIST_CASE(TH3F, TH3)
    CREATE_HIST_CASE(TH3I, TH3)
    CREATE_HIST_CASE(TH3C, TH3)
    CREATE_HIST_CASE(TH3S, TH3)

    CREATE_HIST_CASE(THnD, THn)
    CREATE_HIST_CASE(THnF, THn)
    CREATE_HIST_CASE(THnI, THn)
    CREATE_HIST_CASE(THnC, THn)
    CREATE_HIST_CASE(THnS, THn)
    CREATE_HIST_CASE(THnL, THn)

    CREATE_HIST_CASE(THnSparseD, THnSparse)
    CREATE_HIST_CASE(THnSparseF, THnSparse)
    CREATE_HIST_CASE(THnSparseI, THnSparse)
    CREATE_HIST_CASE(THnSparseC, THnSparse)
    CREATE_HIST_CASE(THnSparseS, THnSparse)
    CREATE_HIST_CASE(THnSparseL, THnSparse)

    CREATE_HIST_CASE(TProfile, TProfile)
    CREATE_HIST_CASE(TProfile2D, TProfile2D)
    CREATE_HIST_CASE(TProfile3D, TProfile3D)

    CREATE_HIST_CASE(StepTHnF, StepTHn)
    CREATE_HIST_CASE(StepTHnD, StepTHn)
    default:
      throw runtime_error("Histogram type was not specified.");
  }
}

// helper function to get the axis via index for any type of root histogram
template <typename T>
TAxis* HistFactory::getAxis(const int i, T* hist)
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

// explicitly instantiate createHist templates for all histogram types
#define EXPIMPL(HType) \
  template std::unique_ptr<HType> HistFactory::createHist<HType>(const HistogramSpec& histSpec);
EXPIMPL(TH1D);
EXPIMPL(TH1F);
EXPIMPL(TH1I);
EXPIMPL(TH1C);
EXPIMPL(TH1S);
EXPIMPL(TH2D);
EXPIMPL(TH2F);
EXPIMPL(TH2I);
EXPIMPL(TH2C);
EXPIMPL(TH2S);
EXPIMPL(TH3D);
EXPIMPL(TH3F);
EXPIMPL(TH3I);
EXPIMPL(TH3C);
EXPIMPL(TH3S);
EXPIMPL(THnD);
EXPIMPL(THnF);
EXPIMPL(THnI);
EXPIMPL(THnC);
EXPIMPL(THnS);
EXPIMPL(THnL);
EXPIMPL(THnSparseD);
EXPIMPL(THnSparseF);
EXPIMPL(THnSparseI);
EXPIMPL(THnSparseC);
EXPIMPL(THnSparseS);
EXPIMPL(THnSparseL);
EXPIMPL(TProfile);
EXPIMPL(TProfile2D);
EXPIMPL(TProfile3D);
EXPIMPL(StepTHnF);
EXPIMPL(StepTHnD)
#undef EXPIMPL

} // namespace o2::framework
