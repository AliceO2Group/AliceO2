// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CalibPedestal.cxx
/// \author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#include <string>
#include <vector>
#include <algorithm>

#include "TObjArray.h"
#include "TFile.h"

#include "TPCBase/ROC.h"
#include "MathUtils/MathBase.h"
#include "TPCCalibration/CalibPulser.h"

using namespace o2::tpc;
using o2::math_utils::math_base::getStatisticsData;
using o2::math_utils::math_base::StatisticsData;

CalibPulser::CalibPulser(PadSubset padSubset)
  : mNbinsT0{200},
    mXminT0{-2},
    mXmaxT0{2},
    mNbinsQtot{200},
    mXminQtot{10},
    mXmaxQtot{40},
    mNbinsWidth{100},
    mXminWidth{0.1},
    mXmaxWidth{5.1},
    mFirstTimeBin{10},
    mLastTimeBin{490},
    mADCMin{5},
    mADCMax{1023},
    mNumberOfADCs{mADCMax - mADCMin + 1},
    mPeakIntMinus{2},
    mPeakIntPlus{2},
    mMinimumQtot{20},
    mT0{"PulserT0"},
    mWidth{"PulserWidth"},
    mQtot{"PulserQtot"},
    mPedestal{nullptr},
    mNoise{nullptr},
    mPulserData{},
    mT0Histograms{},
    mWidthHistograms{},
    mQtotHistograms{}
{
  mT0Histograms.resize(ROC::MaxROC);
  mWidthHistograms.resize(ROC::MaxROC);
  mQtotHistograms.resize(ROC::MaxROC);

  //TODO: until automatic T0 calibration is done we use the same time range
  //      as for the time bin selection
  mXminT0 = mFirstTimeBin;
  mXmaxT0 = mLastTimeBin;
}

//______________________________________________________________________________
Int_t CalibPulser::updateROC(const Int_t roc, const Int_t row, const Int_t pad,
                             const Int_t timeBin, Float_t signal)
{
  // ===| range checks |========================================================
  if (signal < mADCMin || signal > mADCMax)
    return 0;
  if (timeBin < mFirstTimeBin || timeBin > mLastTimeBin)
    return 0;

  // ===| correct the signal |==================================================
  // TODO before or after signal range check?
  if (mPedestal) {
    signal -= mPedestal->getValue(ROC(roc), row, pad);
  }

  // ===| temporary calibration data |==========================================
  const PadROCPos padROCPos(roc, row, pad);
  VectorType& adcData = mPulserData[padROCPos];

  if (!adcData.size()) {
    // accept first and last time bin, so difference +1
    adcData.resize(mLastTimeBin - mFirstTimeBin + 1);
    //printf("new pad pos\n");
  }

  adcData[timeBin - mFirstTimeBin] = signal;
  //printf("%2d, %3d, %3d, %3d: %.2f\n", roc, row, pad, timeBin, signal);
  return 1;
}

//______________________________________________________________________________
void CalibPulser::endReader()
{
  // loop over all signals of all pads filled for the present raw reader
  for (auto& keyValue : mPulserData) {
    const auto& padROCPos = keyValue.first;
    const auto& adcData = keyValue.second;
    const auto currentChannel = mMapper.getPadNumberInROC(padROCPos);

    //std::cout << (int)padROCPos.getROC().getRoc() << ", " << padROCPos.getRow() << ", " << padROCPos.getPad() << std::endl;
    //for (auto& val: adcData) std::cout << val << ", ";
    //std::cout << std::endl;

    const auto data = processPadData(padROCPos, adcData);
    //std::cout << data.mT0+mFirstTimeBin << " : " << data.mQtot << " : " << data.mWidth << "\n";

    // fill histograms
    getHistoT0(padROCPos.getROC(), true)->Fill(data.mT0 + mFirstTimeBin, currentChannel);
    getHistoQtot(padROCPos.getROC(), true)->Fill(data.mQtot, currentChannel);
    getHistoSigma(padROCPos.getROC(), true)->Fill(data.mWidth, currentChannel);
  }

  // reset the adc data to free space
  mPulserData.clear();
}

//______________________________________________________________________________
TH2S* CalibPulser::getHistogram(ROC roc, CalibPulser::PtrVectorType& rocVector,
                                int nbins, float min, float max,
                                std::string_view type, bool create /*=kFALSE*/)
{
  TH2S* vec = rocVector[roc].get();
  if (vec || !create)
    return vec;

  const size_t nChannels = mMapper.getNumberOfPads(roc);
  rocVector[roc] = std::make_unique<TH2S>(Form("hCalib%s%02d", type.data(), roc.getRoc()),
                                          Form("%s calibration histogram ROC %02d", type.data(), roc.getRoc()),
                                          nbins, min, max,
                                          nChannels, 0, nChannels);
  //printf("new histogram %s for ROC %2d\n", type.data(), int(roc));
  return rocVector[roc].get();
}

//______________________________________________________________________________
CalibPulser::PulserData CalibPulser::processPadData(const PadROCPos& padROCPos, const CalibPulser::VectorType& adcData)
{
  // data to return
  PulserData data;

  const auto vectorSize = adcData.size();
  const auto maxElement = std::max_element(std::begin(adcData), std::end(adcData));
  const auto maxPosition = std::distance(std::begin(adcData), maxElement);

  double weightedSum = 0.;
  double weightedSum2 = 0.;
  double chargeSum = 0.;
  for (int t = maxPosition - mPeakIntMinus; t <= maxPosition + mPeakIntPlus; ++t) {
    const auto signal = adcData[t];
    // check time bounds
    if (t < 0 || t >= vectorSize)
      continue;
    weightedSum += signal * (t + 0.5); // +0.5 to get the center of the time bin
    weightedSum2 += signal * (t + 0.5) * (t + 0.5);
    chargeSum += signal;
  }

  if (chargeSum > mMinimumQtot) {
    weightedSum /= chargeSum;
    weightedSum2 = std::sqrt(std::abs(weightedSum2 / chargeSum - weightedSum * weightedSum));
    // L1 phase correction?
    data.mT0 = weightedSum;
    data.mWidth = weightedSum2;
    data.mQtot = chargeSum;
  }

  return data;
}

//______________________________________________________________________________
void CalibPulser::resetData()
{
  mPulserData.clear();

  std::vector<PtrVectorType*> v{&mT0Histograms, &mWidthHistograms, &mQtotHistograms};
  for (auto histArray : v) {
    for (auto& histPtr : *histArray) {
      auto ptr = histPtr.get();
      if (ptr)
        ptr->Reset();
    }
  }
}

//______________________________________________________________________________
void CalibPulser::analyse()
{
  for (ROC roc; !roc.looped(); ++roc) {
    auto histT0 = mT0Histograms.at(roc).get();
    auto histWidth = mWidthHistograms.at(roc).get();
    auto histQtot = mQtotHistograms.at(roc).get();

    if (!histT0 || !histWidth || !histQtot)
      continue;

    // array pointer
    const auto arrT0 = histT0->GetArray();
    const auto arrWidth = histWidth->GetArray();
    const auto arrQtot = histQtot->GetArray();

    const auto numberOfPads = mMapper.getNumberOfPads(roc);
    for (auto iChannel = 0; iChannel < numberOfPads; ++iChannel) {
      const int offsetT0 = (mNbinsT0 + 2) * (iChannel + 1) + 1;
      const int offsetQtot = (mNbinsQtot + 2) * (iChannel + 1) + 1;
      const int offsetWidth = (mNbinsWidth + 2) * (iChannel + 1) + 1;

      StatisticsData dataT0 = getStatisticsData(arrT0 + offsetT0, mNbinsT0, mXminT0, mXmaxT0);
      StatisticsData dataWidth = getStatisticsData(arrWidth + offsetWidth, mNbinsWidth, mXminWidth, mXmaxWidth);
      StatisticsData dataQtot = getStatisticsData(arrQtot + offsetQtot, mNbinsQtot, mXminQtot, mXmaxQtot);

      mT0.getCalArray(roc).setValue(iChannel, dataT0.mCOG);
      mWidth.getCalArray(roc).setValue(iChannel, dataWidth.mCOG);
      mQtot.getCalArray(roc).setValue(iChannel, dataQtot.mCOG);
    }
  }
}

//______________________________________________________________________________
void CalibPulser::dumpToFile(const std::string filename)
{
  auto f = std::unique_ptr<TFile>(TFile::Open(filename.c_str(), "recreate"));
  f->WriteObject(&mT0, "T0");
  f->WriteObject(&mWidth, "Width");
  f->WriteObject(&mQtot, "Qtot");

  if (mDebugLevel) {
    printf("dump debug info\n");
    // temporary arrays for writing the objects
    TObjArray vT0;
    for (auto& val : mT0Histograms)
      vT0.Add(val.get());
    TObjArray vWidth;
    for (auto& val : mWidthHistograms)
      vWidth.Add(val.get());
    TObjArray vQtot;
    for (auto& val : mQtotHistograms)
      vQtot.Add(val.get());

    vT0.Write("T0Histograms", TObject::kSingleKey);
    vWidth.Write("WidthHistograms", TObject::kSingleKey);
    vQtot.Write("QtotHistograms", TObject::kSingleKey);
  }

  f->Close();
}
