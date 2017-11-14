// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TH1.h>

#include "QCProducer/TH1Producer.h"

namespace o2
{
namespace qc
{
TH1Producer::TH1Producer(const char* histogramName, const char* histogramTitle, const int numberOfbins)
  : mBeansNumber(numberOfbins)
{
  mHistogramName = histogramName;
  mHistogramTitle = histogramTitle;
}

TObject* TH1Producer::produceData() const
{
  auto* histogram = new TH1F(mHistogramName.c_str(), mHistogramTitle.c_str(), mBeansNumber, mXLow, mXUp);
  histogram->FillRandom("gaus", 1000);

  return histogram;
}
}
}
