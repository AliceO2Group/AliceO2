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

#include "DataFormatsFT0/GlobalOffsetsContainer.h"
#include <numeric>
#include <algorithm>
#include "MathUtils/fit.h"
#include <TFitResult.h>
#include <gsl/span>

using namespace o2::ft0;

GlobalOffsetsContainer::GlobalOffsetsContainer(std::size_t minEntries)
  : mMinEntries(minEntries)
{
  TString histnames[3] = {"hT0A", "hT0C", "hT0AC"};
  for (int ihist = 0; ihist < 3; ++ihist) {
    mHistogram[ihist] = new TH1F(histnames[ihist].Data(), histnames[ihist].Data(), NUMBER_OF_HISTOGRAM_BINS, -HISTOGRAM_RANGE, HISTOGRAM_RANGE);
  }
}

bool GlobalOffsetsContainer::hasEnoughEntries() const
{
  LOG(debug) << "@@@ GlobalOffsetsContainer::hasEnoughEntries " << *std::min_element(mEntriesCollTime.begin(), mEntriesCollTime.end());
  return *std::min_element(mEntriesCollTime.begin(), mEntriesCollTime.end()) > mMinEntries;
}
void GlobalOffsetsContainer::fill(const gsl::span<const o2::ft0::RecoCalibInfoObject>& data)
{
  for (auto& entry : data) {
    if (std::abs(entry.getT0A()) < HISTOGRAM_RANGE && std::abs(entry.getT0C()) < HISTOGRAM_RANGE && std::abs(entry.getT0AC()) < HISTOGRAM_RANGE) {
      mHistogram[0]->Fill(entry.getT0A());
      mHistogram[1]->Fill(entry.getT0C());
      mHistogram[2]->Fill(entry.getT0AC());
      ++mEntriesCollTime[0];
      ++mEntriesCollTime[1];
      ++mEntriesCollTime[2];
    } else {
      LOG(debug) << "empty  data A " << entry.getT0A() << " C " << entry.getT0C();
    }
  }
}

void GlobalOffsetsContainer::merge(GlobalOffsetsContainer* prev)
{
  LOG(debug) << "@@@ GlobalOffsetsContainer::merge";
  for (int ihist = 0; ihist < 3; ++ihist) {
    mHistogram[ihist]->Add(prev->mHistogram[ihist], mHistogram[ihist], 1, 1);
  }
}

int16_t GlobalOffsetsContainer::getMeanGaussianFitValue(std::size_t side) const
{

  //  static constexpr size_t MEAN_VALUE_INDEX_IN_OUTPUT_VECTOR = 1;

  if (0 == mHistogram[side]->GetEntries()) {
    return 0;
  }

  TFitResultPtr returnCode = mHistogram[side]->Fit("gaus", "SQ");
  if (returnCode < 0) {
    LOG(error) << "Gaussian fit error!";
    return 0;
  }
  double meanfit = 0;
  if ((Int_t)returnCode == 0) {
    meanfit = returnCode->Parameter(1);
  }
  LOG(debug) << " @@@ MeanGaussianFitValue " << meanfit;
  return meanfit;
}

void GlobalOffsetsContainer::print() const
{
  for (int ihist = 0; ihist < 3; ++ihist) {
    LOG(info) << "Container keep data for global offsets calibration:";
    LOG(info) << "Gaussian mean time A side " << getMeanGaussianFitValue(0) << " based on" << mEntriesCollTime[0] << " entries";
    LOG(info) << "Gaussian mean time C side " << getMeanGaussianFitValue(1) << " based on" << mEntriesCollTime[1] << " entries";
    LOG(info) << "Gaussian mean time AC side " << getMeanGaussianFitValue(2) << " based on" << mEntriesCollTime[2] << " entries";
  }
}
