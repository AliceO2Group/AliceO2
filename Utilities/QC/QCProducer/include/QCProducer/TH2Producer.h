// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#pragma once

#include <Rtypes.h>

#include "Producer.h"
#include <string>

namespace o2
{
namespace qc
{
class TH2Producer : public Producer
{
 public:
  TH2Producer(const char* histogramName, const char* histogramTitle, const int bins);
  TObject* produceData() const override;

 private:
  std::string mHistogramName;
  std::string mHistogramTitle;

  const Int_t mNbinsx;
  const Int_t mNbinsy;
  const Double_t mXlow{ -10 };
  const Double_t mXup{ 10 };
  const Double_t mYlow{ -20 };
  const Double_t mYup{ 20 };
};
}
}
