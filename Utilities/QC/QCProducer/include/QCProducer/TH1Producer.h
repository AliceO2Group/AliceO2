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

#include "Producer.h"
#include <string>

namespace o2
{
namespace qc
{
class TH1Producer : public Producer
{
 public:
  TH1Producer(const char* histogramName, const char* histogramTitle, const int numberOfBins);
  TObject* produceData() const override;

 private:
  std::string mHistogramName;
  std::string mHistogramTitle;

  const int mBeansNumber;
  const double mXLow{ -10 };
  const double mXUp{ 10 };
  const float mNumberOfEntries{ 1000 };
};
}
}
