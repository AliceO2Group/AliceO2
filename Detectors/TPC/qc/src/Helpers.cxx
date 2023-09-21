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

#include <cassert>

// o2 includes
#include "TPCQC/Helpers.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/ROC.h"

using namespace o2::tpc::qc;

//______________________________________________________________________________
std::vector<double> helpers::makeLogBinning(const int nbins, const double min, const double max)
{
  assert(min > 0);
  assert(min < max);

  std::vector<double> binLim(nbins + 1);

  const double expMax = std::log(max / min);
  const double binWidth = expMax / nbins;

  binLim[0] = min;
  binLim[nbins] = max;

  for (Int_t i = 1; i < nbins; ++i) {
    binLim[i] = min * std::exp(i * binWidth);
  }

  return binLim;
}

//______________________________________________________________________________
void helpers::setStyleHistogram1D(TH1& histo)
{
  histo.SetStats(1);
}

//______________________________________________________________________________
void helpers::setStyleHistogram1D(std::vector<TH1F>& histos)
{
  for (auto& hist : histos) {
    helpers::setStyleHistogram1D(hist);
  }
}

//______________________________________________________________________________
void helpers::setStyleHistogram2D(TH2& histo)
{
  histo.SetOption("colz");
  histo.SetStats(0);
  histo.SetMinimum(0.9);
}

//______________________________________________________________________________
void helpers::setStyleHistogram2D(std::vector<TH2F>& histos)
{
  for (auto& hist : histos) {
    helpers::setStyleHistogram2D(hist);
  }
}

//______________________________________________________________________________
void helpers::setStyleHistogram(TH1& histo)
{
  // common and 1D case
  histo.SetStats(1);
  // 2D case
  if (histo.InheritsFrom(TH2::Class())) {
    histo.SetOption("colz");
    histo.SetStats(0);
    histo.SetMinimum(0.9);
  }
}

//______________________________________________________________________________
void helpers::setStyleHistogramsInMap(std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>>& mapOfvectors)
{
  for (const auto& keyValue : mapOfvectors) {
    for (auto& hist : keyValue.second) {
      helpers::setStyleHistogram(*hist);
    }
  }
}

//______________________________________________________________________________
void helpers::setStyleHistogramsInMap(std::unordered_map<std::string_view, std::unique_ptr<TH1>>& mapOfHisto)
{
  for (const auto& keyValue : mapOfHisto) {
    helpers::setStyleHistogram(*(keyValue.second));
  }
}

//______________________________________________________________________________
void helpers::setStyleHistogramsInMap(std::unordered_map<std::string, std::vector<std::unique_ptr<TH1>>>& mapOfvectors)
{
  for (const auto& keyValue : mapOfvectors) {
    for (auto& hist : keyValue.second) {
      helpers::setStyleHistogram(*hist);
    }
  }
}

//______________________________________________________________________________
void helpers::setStyleHistogramsInMap(std::unordered_map<std::string, std::unique_ptr<TH1>>& mapOfHisto)
{
  for (const auto& keyValue : mapOfHisto) {
    helpers::setStyleHistogram(*(keyValue.second));
  }
}

//______________________________________________________________________________
bool helpers::newZSCalib(const o2::tpc::CalDet<float>& refPedestal, const o2::tpc::CalDet<float>& refNoise, const o2::tpc::CalDet<float>& pedestal)
{
  const o2::tpc::Mapper& mapper = o2::tpc::Mapper::instance();
  o2::tpc::CalDet<float> diffCalDet = refPedestal - pedestal;

  for (o2::tpc::ROC roc; !roc.looped(); ++roc) {
    const int nrows = mapper.getNumberOfRowsROC(roc);
    for (int irow = 0; irow < nrows; ++irow) {
      const int npads = mapper.getNumberOfPadsInRowROC(roc, irow);
      for (int ipad = 0; ipad < npads; ++ipad) {
        const auto val = diffCalDet.getValue(roc, irow, ipad);
        if (std::abs(val) > 3 * refNoise.getValue(roc, irow, ipad)) {
          return true;
        }
      }
    }
  }

  return false;
}
