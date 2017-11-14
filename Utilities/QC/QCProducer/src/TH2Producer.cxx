// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "QCProducer/TH2Producer.h"

#include <TF2.h>
#include <TH2F.h>

using namespace std;

namespace o2
{
namespace qc
{
TH2Producer::TH2Producer(const char* histogramName, const char* histogramTitle, const int bins)
  : mHistogramName(histogramName), mHistogramTitle(histogramTitle), mNbinsx(bins), mNbinsy(bins)
{
}

TObject* TH2Producer::produceData() const
{
  auto* histogram = new TH2F(mHistogramName.c_str(), mHistogramTitle.c_str(), mNbinsx, mXlow, mXup, mNbinsy, mYlow, mYup);

  for (int i = 0; i < mNbinsx; ++i) {
    for (int j = 0; j < mNbinsy; ++j) {
      histogram->Fill(i, j, 1.0);
    }
  }

  return histogram;
}
}
}
