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

/// \file   CalibPedestal.cxx
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <fmt/format.h>

#include "TH2F.h"
#include "TFile.h"

#include "TPCBase/ROC.h"
#include "MathUtils/fit.h"
#include "TPCCalibration/CalibPedestal.h"

using namespace o2::tpc;
using o2::math_utils::fit;
using o2::math_utils::fitGaus;
using o2::math_utils::getStatisticsData;
using o2::math_utils::StatisticsData;

CalibPedestal::CalibPedestal(PadSubset padSubset)
  : CalibRawBase(padSubset),
    mFirstTimeBin(0),
    mLastTimeBin(500),
    mADCMin(20),
    mADCMax(140),
    mNumberOfADCs(mADCMax - mADCMin + 1),
    mStatisticsType(StatisticsType::GausFitFast),
    mPedestal("Pedestals", padSubset),
    mNoise("Noise", padSubset),
    mADCdata()

{
  mADCdata.resize(ROC::MaxROC);
}
//______________________________________________________________________________
void CalibPedestal::init()
{
  const auto& param = CalibPedestalParam::Instance();

  mFirstTimeBin = param.FirstTimeBin;
  mLastTimeBin = param.LastTimeBin;
  mADCMin = param.ADCMin;
  mADCMax = param.ADCMax;
  mNumberOfADCs = mADCMax - mADCMin + 1;
  mStatisticsType = param.StatType;
}

//______________________________________________________________________________
Int_t CalibPedestal::updateROC(const Int_t roc, const Int_t row, const Int_t pad,
                               const Int_t timeBin, const Float_t signal)
{
  Int_t adcValue = Int_t(signal);
  if (timeBin < mFirstTimeBin || timeBin > mLastTimeBin) {
    return 0;
  }
  if (adcValue < mADCMin || adcValue > mADCMax) {
    return 0;
  }

  const GlobalPadNumber padInROC = mMapper.getPadNumberInROC(PadROCPos(roc, row, pad));
  Int_t bin = padInROC * mNumberOfADCs + (adcValue - mADCMin);
  vectorType& adcVec = *getVector(ROC(roc), kTRUE);
  ++(adcVec[bin]);

  //printf("bin: %5d, val: %.2f\n", bin, adcVec[bin]);

  return 0;
}

//______________________________________________________________________________
CalibPedestal::vectorType* CalibPedestal::getVector(ROC roc, bool create /*=kFALSE*/)
{
  vectorType* vec = mADCdata[roc].get();
  if (vec || !create) {
    return vec;
  }

  const size_t numberOfPads = (roc.rocType() == RocType::IROC) ? mMapper.getPadsInIROC() : mMapper.getPadsInOROC();

  vec = new vectorType;
  vec->resize(numberOfPads * mNumberOfADCs);

  mADCdata[roc] = std::unique_ptr<vectorType>(vec);

  return vec;
}

//______________________________________________________________________________
void CalibPedestal::analyse()
{
  ROC roc;

  std::vector<float> fitValues;

  for (auto& vecPtr : mADCdata) {
    auto vec = vecPtr.get();
    if (!vec) {
      ++roc;
      continue;
    }

    CalROC& calROCPedestal = mPedestal.getCalArray(roc);
    CalROC& calROCNoise = mNoise.getCalArray(roc);

    float* array = vec->data();

    const size_t numberOfPads = (roc.rocType() == RocType::IROC) ? mMapper.getPadsInIROC() : mMapper.getPadsInOROC();

    float pedestal{};
    float noise{};

    TF1 fg("fg", "gaus");
    fg.SetRange(mADCMin - 0.5f, mADCMax + 1.5f);

    for (Int_t ichannel = 0; ichannel < numberOfPads; ++ichannel) {
      size_t offset = ichannel * mNumberOfADCs;
      if (mStatisticsType == StatisticsType::GausFit) {
        fit(mNumberOfADCs, array + offset, float(mADCMin) - 0.5f, float(mADCMax + 1) - 0.5f, fg); // -0.5 since ADC values are discrete
        pedestal = fg.GetParameter(1);
        noise = fg.GetParameter(2);
      } else if (mStatisticsType == StatisticsType::GausFitFast) {
        fitGaus(mNumberOfADCs, array + offset, float(mADCMin) - 0.5f, float(mADCMax + 1) - 0.5f, fitValues); // -0.5 since ADC values are discrete
        pedestal = fitValues[1];
        noise = fitValues[2];
      } else if (mStatisticsType == StatisticsType::MeanStdDev) {
        StatisticsData data = getStatisticsData(array + offset, mNumberOfADCs, double(mADCMin) - 0.5, double(mADCMax) - 0.5); // -0.5 since ADC values are discrete
        pedestal = data.mCOG;
        noise = data.mStdDev;
      }
      noise = std::abs(noise); // noise can be negative in gaus fit

      calROCPedestal.setValue(ichannel, pedestal);
      calROCNoise.setValue(ichannel, noise);

      //printf("roc: %2d, channel: %4d, pedestal: %.2f, noise: %.2f\n", roc.getRoc(), ichannel, pedestal, noise);
    }

    ++roc;
  }
}

//______________________________________________________________________________
void CalibPedestal::resetData()
{
  for (auto& vecPtr : mADCdata) {
    auto vec = vecPtr.get();
    if (!vec) {
      continue;
    }
    vec->clear();
  }
}

//______________________________________________________________________________
void CalibPedestal::dumpToFile(const std::string filename, uint32_t type /* = 0*/)
{
  auto f = std::unique_ptr<TFile>(TFile::Open(filename.c_str(), "recreate"));
  if (type == 0) {
    f->WriteObject(&mPedestal, "Pedestals");
    f->WriteObject(&mNoise, "Noise");
    f->Close();
  } else if (type == 1) {
    f->WriteObject(this, "CalibPedestal");
  }
}

//______________________________________________________________________________
TH2* CalibPedestal::createControlHistogram(ROC roc)
{
  auto* data = mADCdata[roc.getRoc()]->data();

  const size_t numberOfPads = (roc.rocType() == RocType::IROC) ? mMapper.getPadsInIROC() : mMapper.getPadsInOROC();
  TH2F* h2 = new TH2F(fmt::format("hADCValues_ROC{:02}", roc.getRoc()).data(), fmt::format("ADC values of ROC {:02}", roc.getRoc()).data(), numberOfPads, 0, numberOfPads, mNumberOfADCs, mADCMin, mADCMax);
  h2->SetDirectory(nullptr);
  for (int ichannel = 0; ichannel < numberOfPads; ++ichannel) {
    size_t offset = ichannel * mNumberOfADCs;

    for (int iADC = 0; iADC < mNumberOfADCs; ++iADC) {
      h2->Fill(ichannel, mADCMin + iADC, (data + offset)[iADC]);
    }
  }

  return h2;
}
