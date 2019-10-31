// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cassert>

//root includes
#include "TH1F.h"
#include "TH2F.h"

//o2 includes
#include "TPCQC/Helpers.h"

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
