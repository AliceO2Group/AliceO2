// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "QCProducer/TH3Producer.h"

#include <TH3.h>
#include <TRandom.h>

using namespace std;

namespace o2
{
namespace qc
{
TH3Producer::TH3Producer(const char* histogramName, const char* histogramTitle, const int bins)
  : mHistogramName(histogramName), mHistogramTitle(histogramTitle), mNbinsx(bins), mNbinsy(bins), mNbinsz(bins)
{
}

TObject* TH3Producer::produceData() const
{
  auto* histogram =
    new TH3F(mHistogramName.c_str(), mHistogramTitle.c_str(), mNbinsx, mXlow, mXup, mNbinsy, mYlow, mYup, mNbinsz, mZlow, mZup);

  Double_t x, y, z;

  for (Int_t i = 0; i < 1000; ++i) {
    gRandom->Rannor(x, y);
    z = x * x + y * y;
    histogram->Fill(x, y, z);
  }

  return histogram;
}
}
}
