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

/// \file  TPCScaler.cxx
/// \brief Definition of TPCScaler class
///
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>

#include "TPCCalibration/TPCScaler.h"
#include <TFile.h>
#include <TTree.h>
#include "Framework/Logger.h"
#include "CommonConstants/LHCConstants.h"

using namespace o2::tpc;

void TPCScaler::dumpToFile(const char* file, const char* name)
{
  TFile out(file, "RECREATE");
  TTree tree(name, name);
  tree.SetAutoSave(0);
  tree.Branch("TPCScaler", this);
  tree.Fill();
  out.WriteObject(&tree, name);
}

void TPCScaler::dumpToFile(const char* file, const char* name, double startTimeMS, double endTimeMS)
{
  TPCScaler scaler;
  scaler.mIonDriftTimeMS = mIonDriftTimeMS;
  scaler.mRun = mRun;
  scaler.mFirstTFOrbit = (startTimeMS <= 0) ? mFirstTFOrbit : (mFirstTFOrbit + (startTimeMS - mTimeStampMS) / (o2::constants::lhc::LHCOrbitMUS * 0.001));
  scaler.mTimeStampMS = (startTimeMS <= 0) ? mTimeStampMS : startTimeMS;
  scaler.mIntegrationTimeMS = mIntegrationTimeMS;

  const int dataIdx = (startTimeMS <= 0) ? 0 : getDataIdx(startTimeMS);
  const int idxDataStart = (startTimeMS <= 0) ? 0 : dataIdx;
  const int idxDataEndA = ((endTimeMS < 0) || (dataIdx > getNValues(Side::A))) ? getNValues(Side::A) : dataIdx;
  const int idxDataEndC = ((endTimeMS < 0) || (dataIdx > getNValues(Side::C))) ? getNValues(Side::C) : dataIdx;
  scaler.mScalerA = std::vector<float>(mScalerA.begin() + idxDataStart, mScalerA.begin() + idxDataEndA);
  scaler.mScalerC = std::vector<float>(mScalerC.begin() + idxDataStart, mScalerC.begin() + idxDataEndC);
  scaler.dumpToFile(file, name);
}

void TPCScaler::dumpToFileSlices(const char* file, const char* name, int minutesPerObject, float marginMS, float marginCCDBMinutes)
{
  if (getNValues(o2::tpc::Side::A) != getNValues(o2::tpc::Side::C)) {
    LOGP(error, "Number of points stored for A-side and C-side is different");
    return;
  }
  const long marginCCDBMS = marginCCDBMinutes * 60 * 1000;
  const float msPerObjectTmp = minutesPerObject * 60 * 1000;
  const int valuesPerSlice = static_cast<int>(msPerObjectTmp / mIntegrationTimeMS);
  const int marginPerSlice = static_cast<int>(marginMS / mIntegrationTimeMS);
  int nSlices = getNValues(Side::A) / valuesPerSlice; // number of output objects

  LOGP(info, "Producing {} objects with a CCDB margin of {} ms and {} margin per slice", nSlices, marginCCDBMS, marginPerSlice);
  if (nSlices == 0) {
    nSlices = 1;
  }

  for (int i = 0; i < nSlices; ++i) {
    const int idxDataStart = (i == 0) ? 0 : (i * valuesPerSlice - marginPerSlice);
    int idxDataEnd = (i == nSlices - 1) ? getNValues(Side::A) : ((i + 1) * valuesPerSlice + marginPerSlice);
    if (idxDataEnd > getNValues(Side::A)) {
      idxDataEnd = getNValues(Side::A);
    }

    TPCScaler scaler;
    scaler.mIonDriftTimeMS = mIonDriftTimeMS;
    scaler.mRun = mRun;
    scaler.mTimeStampMS = (i == 0) ? mTimeStampMS : (mTimeStampMS + idxDataStart * static_cast<double>(mIntegrationTimeMS));
    scaler.mFirstTFOrbit = mFirstTFOrbit + (scaler.mTimeStampMS - mTimeStampMS) / (o2::constants::lhc::LHCOrbitMUS * 0.001);
    scaler.mIntegrationTimeMS = mIntegrationTimeMS;
    scaler.mScalerA = std::vector<float>(mScalerA.begin() + idxDataStart, mScalerA.begin() + idxDataEnd);
    scaler.mScalerC = std::vector<float>(mScalerC.begin() + idxDataStart, mScalerC.begin() + idxDataEnd);

    const long timePerSliceMS = valuesPerSlice * mIntegrationTimeMS;
    const long tsCCDBStart = mTimeStampMS + i * timePerSliceMS;
    const long tsCCDBStartMargin = (i == 0) ? (tsCCDBStart - marginCCDBMS) : tsCCDBStart;
    const long tsCCDBEnd = (i == nSlices - 1) ? (getEndTimeStampMS(o2::tpc::Side::A) + marginCCDBMS) : (tsCCDBStart + timePerSliceMS);
    const std::string fileOut = fmt::format("{}_{}_{}_{}.root", file, i, tsCCDBStartMargin, tsCCDBEnd);
    scaler.dumpToFile(fileOut.data(), name);
  }
}

void TPCScaler::loadFromFile(const char* inpf, const char* name)
{
  TFile out(inpf, "READ");
  TTree* tree = (TTree*)out.Get(name);
  setFromTree(*tree);
}

void TPCScaler::setFromTree(TTree& tpcScalerTree)
{
  TPCScaler* scalerTmp = this;
  tpcScalerTree.SetBranchAddress("TPCScaler", &scalerTmp);
  const int entries = tpcScalerTree.GetEntries();
  if (entries > 0) {
    tpcScalerTree.GetEntry(0);
  } else {
    LOGP(error, "TPCScaler not found in input file");
  }
  tpcScalerTree.SetBranchAddress("TPCScaler", nullptr);
}

float TPCScaler::getMeanScaler(double timestamp, o2::tpc::Side side) const
{
  // index to data buffer
  const int idxData = getDataIdx(timestamp);
  const int nVals = getNValuesIonDriftTime();
  const int nValues = getNValues(side);
  if ((nVals == 0) || (nVals > nValues)) {
    return -1;
    LOGP(error, "Empty data provided {}", nValues);
  }

  // clamp indices to min and max
  const int lastIdx = std::clamp(idxData, nVals, nValues);
  const int firstIdx = (lastIdx == nValues) ? (nValues - nVals) : std::clamp(idxData - nVals, 0, nValues);

  // sump up values from last ion drift time
  float sum = 0;
  float sumW = 0;
  const bool useWeights = mUseWeights && getScalerWeights().isValid();
  for (int i = firstIdx; i < lastIdx; ++i) {
    float weight = 1;
    if (useWeights) {
      const double relTSMS = mTimeStampMS + i * mIntegrationTimeMS - timestamp;
      weight = getScalerWeights().getWeight(relTSMS);
    }
    sum += getScalers(i, side) * weight;
    sumW += weight;
  }
  if (sumW != 0) {
    return (sum / sumW);
  }
  return 0;
}

float TPCScalerWeights::getWeight(float deltaTime) const
{
  const float idxF = (deltaTime - mFirstTimeStampMS) / mSamplingTimeMS;
  const int idx = idxF;
  if ((idx < 0) || (idx > mWeights.size() - 1)) {
    LOGP(error, "Index out of range for deltaTime: {} mFirstTimeStampMS: {} mSamplingTimeMS: {}", deltaTime, mFirstTimeStampMS, mSamplingTimeMS);
    // set weight 1 to in case it is out of range. This can only happen if the TPC scaler is not valid for given time
    return 1;
  }

  if ((idxF == idx) || (idx == mWeights.size() - 1)) {
    // no interpolation required
    return mWeights[idx];
  } else {
    // interpolate scaler
    const float y0 = mWeights[idx];
    const float y1 = mWeights[idx + 1];
    const float x0 = idx;
    const float x1 = idx + 1;
    const float x = idxF;
    const float y = ((y0 * (x1 - x)) + y1 * (x - x0)) / (x1 - x0);
    return y;
  }
}

void TPCScaler::clampScalers(float minThreshold, float maxThreshold, Side side)
{
  auto& scaler = (side == o2::tpc::Side::A) ? mScalerA : mScalerC;
  std::transform(std::begin(scaler), std::end(scaler), std::begin(scaler), [minThreshold, maxThreshold](auto val) { return std::clamp(val, minThreshold, maxThreshold); });
}
