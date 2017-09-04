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

#include "TFile.h"
#include "TPCBase/ROC.h"
#include "MathUtils/MathBase.h"
#include "TPCCalibration/CalibPedestal.h"

using namespace o2::TPC;
using o2::mathUtils::mathBase::fitGaus;

CalibPedestal::CalibPedestal(PadSubset padSubset)
  : CalibRawBase(padSubset),
    mADCMin(0),
    mADCMax(120),
    mNumberOfADCs(mADCMax-mADCMin+1),
    mPedestal(padSubset),
    mNoise(padSubset),
    mADCdata()

{
  mADCdata.resize(ROC::MaxROC);
  mPedestal.setName("Pedestals");
  mNoise.setName("Noise");
}

//______________________________________________________________________________
Int_t CalibPedestal::updateROC(const Int_t roc, const Int_t row, const Int_t pad,
                               const Int_t timeBin, const Float_t signal)
{
  Int_t adcValue = Int_t(signal);
  if (adcValue<mADCMin || adcValue>mADCMax) return 0;

  const GlobalPadNumber padInROC = mMapper.getPadNumberInROC(PadROCPos(roc, row, pad));
  Int_t bin = padInROC * mNumberOfADCs + (adcValue-mADCMin);
  vectorType& adcVec = *getVector(ROC(roc), kTRUE);
  ++(adcVec[bin]);

  //printf("bin: %5d, val: %.2f\n", bin, adcVec[bin]);

  return 0;
}

//______________________________________________________________________________
CalibPedestal::vectorType* CalibPedestal::getVector(ROC roc, bool create/*=kFALSE*/)
{
  vectorType* vec = mADCdata[roc].get();
  if (vec || !create) return vec;

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

    float *array = vec->data();

    const size_t numberOfPads = (roc.rocType() == RocType::IROC) ? mMapper.getPadsInIROC() : mMapper.getPadsInOROC();

    for (Int_t ichannel=0; ichannel<numberOfPads; ++ichannel) {
      size_t offset = ichannel * mNumberOfADCs;
      fitGaus(mNumberOfADCs, array+offset, float(mADCMin), float(mADCMax+1), fitValues);

      const float pedestal = fitValues[1];
      const float noise    = fitValues[2];
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
void CalibPedestal::dumpToFile(TString filename)
{
  TFile *f = TFile::Open(filename, "recreate");
  f->WriteObject(&mPedestal, "Pedestals");
  f->WriteObject(&mNoise, "Noise");
  f->Close();
  delete f;
}
