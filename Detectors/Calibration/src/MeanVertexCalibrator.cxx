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

#include "DetectorsCalibration/MeanVertexCalibrator.h"
#include "Framework/Logger.h"
#include "MathUtils/fit.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"

namespace o2
{
namespace calibration
{

using Slot = o2::calibration::TimeSlot<o2::calibration::MeanVertexData>;
using o2::math_utils::fitGaus;
using clbUtils = o2::calibration::Utils;
using MeanVertexObject = o2::dataformats::MeanVertexObject;

void MeanVertexCalibrator::initOutput()
{
  // Here we initialize the vector of our output objects
  mMeanVertexVector.clear();
  mInfoVector.clear();
  return;
}

//_____________________________________________
void MeanVertexCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  o2::calibration::MeanVertexData* c = slot.getContainer();
  LOG(info) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd() << " with "
            << c->getEntries() << " entries";
  mTmpMVobjDqTime.emplace_back(slot.getStartTimeMS(), slot.getEndTimeMS());

  if (mUseFit) {
    MeanVertexObject mvo;
    // x coordinate
    std::vector<float> fitValues;
    float* array = &c->histoX[0];
    if (mVerbose) {
      LOG(info) << "**** Printing content of MeanVertex object for x coordinate";
      for (int i = 0; i < c->nbinsX; i++) {
        LOG(info) << "i = " << i << ", content of histogram = " << c->histoX[i];
      }
    }
    double fitres = fitGaus(c->nbinsX, array, -(c->rangeX), c->rangeX, fitValues);
    if (fitres >= 0) {
      LOG(info) << "X: Fit result (of single Slot) => " << fitres << ". Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(warning) << "X: Fit failed with result = " << fitres;
    }
    mvo.setX(fitValues[1]);
    mvo.setSigmaX(fitValues[2]);

    // y coordinate
    array = &c->histoY[0];
    if (mVerbose) {
      LOG(info) << "**** Printing content of MeanVertex object for y coordinate";
      for (int i = 0; i < c->nbinsY; i++) {
        LOG(info) << "i = " << i << ", content of histogram = " << c->histoY[i];
      }
    }
    fitres = fitGaus(c->nbinsY, array, -(c->rangeY), c->rangeY, fitValues);
    if (fitres >= 0) {
      LOG(info) << "Y: Fit result (of single Slot) => " << fitres << ". Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(warning) << "Y: Fit failed with result = " << fitres;
    }
    mvo.setY(fitValues[1]);
    mvo.setSigmaY(fitValues[2]);

    // z coordinate
    array = &c->histoZ[0];
    if (mVerbose) {
      LOG(info) << "**** Printing content of MeanVertex object for z coordinate";
      for (int i = 0; i < c->nbinsZ; i++) {
        LOG(info) << "i = " << i << ", content of histogram = " << c->histoZ[i];
      }
    }
    fitres = fitGaus(c->nbinsZ, array, -(c->rangeZ), c->rangeZ, fitValues);
    if (fitres >= 0) {
      LOG(info) << "Z: Fit result (of single Slot) => " << fitres << ". Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(warning) << "Z: Fit failed with result = " << fitres;
    }
    mvo.setZ(fitValues[1]);
    mvo.setSigmaZ(fitValues[2]);

    // now we add the object to the deque
    mTmpMVobjDq.push_back(std::move(mvo));
  } else {
    mTmpMVdataDq.push_back(std::move(*c));
    mSMAdata.merge(&mTmpMVdataDq.back());
    if (mTmpMVobjDqTime.size() > mSMAslots) {
      mSMAdata.subtract(&mTmpMVdataDq.front());
      mTmpMVdataDq.pop_front();
    }
  }

  // output object
  MeanVertexObject mvo;

  if (mUseFit) {
    doSimpleMovingAverage(mTmpMVobjDq, mSMAMVobj);
  } else {
    // now we need to fit, on the merged data
    if (mVerbose) {
      LOG(info) << "**** Printing content of SMA MVData object for x coordinate";
      for (int i = 0; i < mSMAdata.nbinsX; i++) {
        LOG(info) << "i = " << i << ", content of histogram = " << mSMAdata.histoX[i];
      }
    }
    std::vector<float> fitValues;
    float* array = &mSMAdata.histoX[0];
    double fitres = fitGaus(mSMAdata.nbinsX, array, -(mSMAdata.rangeX), mSMAdata.rangeX, fitValues);
    if (fitres >= 0) {
      LOG(info) << "X: Fit result (of merged Slots) => " << fitres << ". Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(warning) << "X: Fit failed with result = " << fitres;
    }
    mSMAMVobj.setX(fitValues[1]);
    mSMAMVobj.setSigmaX(fitValues[2]);

    // y coordinate
    if (mVerbose) {
      LOG(info) << "**** Printing content of SMA MVData object for y coordinate";
      for (int i = 0; i < mSMAdata.nbinsY; i++) {
        LOG(info) << "i = " << i << ", content of histogram = " << mSMAdata.histoY[i];
      }
    }
    array = &mSMAdata.histoY[0];
    fitres = fitGaus(mSMAdata.nbinsY, array, -(mSMAdata.rangeY), mSMAdata.rangeY, fitValues);
    if (fitres >= 0) {
      LOG(info) << "Y: Fit result (of merged Slots) => " << fitres << ". Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(warning) << "Y: Fit failed with result = " << fitres;
    }
    mSMAMVobj.setY(fitValues[1]);
    mSMAMVobj.setSigmaY(fitValues[2]);

    // z coordinate
    if (mVerbose) {
      LOG(info) << "**** Printing content of SMA MVData object for z coordinate";
      for (int i = 0; i < mSMAdata.nbinsZ; i++) {
        LOG(info) << "i = " << i << ", content of histogram = " << mSMAdata.histoZ[i];
      }
    }
    array = &mSMAdata.histoZ[0];
    fitres = fitGaus(mSMAdata.nbinsZ, array, -(mSMAdata.rangeZ), mSMAdata.rangeZ, fitValues);
    if (fitres >= 0) {
      LOG(info) << "Z: Fit result (of merged Slots) => " << fitres << ". Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(warning) << "Z: Fit failed with result = " << fitres;
    }
    mSMAMVobj.setZ(fitValues[1]);
    mSMAMVobj.setSigmaZ(fitValues[2]);
  }

  if (mTmpMVobjDqTime.size() > mSMAslots) {
    mTmpMVobjDqTime.pop_front();
  }
  long startValidity = (mTmpMVobjDqTime.front().getMin() + mTmpMVobjDqTime.back().getMax()) / 2;
  LOG(info) << "start validity = " << startValidity;
  std::map<std::string, std::string> md;
  auto clName = o2::utils::MemFileHelper::getClassName(mSMAMVobj);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfoVector.emplace_back("GLO/Calib/MeanVertex", clName, flName, md, startValidity - 10 * o2::ccdb::CcdbObjectInfo::SECOND, startValidity + o2::ccdb::CcdbObjectInfo::MONTH);
  mMeanVertexVector.emplace_back(mSMAMVobj);

  slot.print();
}

//_____________________________________________
void MeanVertexCalibrator::doSimpleMovingAverage(std::deque<float>& dq, float& sma)
{

  // doing simple moving average

  if (dq.size() <= mSMAslots) {
    sma = std::accumulate(dq.begin(), dq.end(), 0.0) / dq.size();
    //avg = (avg * (vect.size() - 1) + vect.back()) / vect.size();
    return;
  }

  // if the vector has size > mSMAslots, we calculate the SMA, and then we drop 1 element
  // (note that it can have a size > mSMAslots only of 1 element!)
  sma += (dq[dq.size() - 1] - dq[0]) / mSMAslots;
  dq.pop_front();

  return;
}

//_____________________________________________
void MeanVertexCalibrator::doSimpleMovingAverage(std::deque<MVObject>& dq, MVObject& sma)
{

  // doing simple moving average

  if (dq.size() <= mSMAslots) {
    sma.setX((sma.getX() * (dq.size() - 1) + dq.back().getX()) / dq.size());
    sma.setY((sma.getY() * (dq.size() - 1) + dq.back().getY()) / dq.size());
    sma.setZ((sma.getZ() * (dq.size() - 1) + dq.back().getZ()) / dq.size());
    sma.setSigmaX((sma.getSigmaX() * (dq.size() - 1) + dq.back().getSigmaX()) / dq.size());
    sma.setSigmaY((sma.getSigmaY() * (dq.size() - 1) + dq.back().getSigmaY()) / dq.size());
    sma.setSigmaZ((sma.getSigmaZ() * (dq.size() - 1) + dq.back().getSigmaZ()) / dq.size());
    return;
  }

  // if the vector has size > mSMAslots, we calculate the SMA, and then we drop 1 element
  // (note that it can have a size > mSMAslots only of 1 element!)
  sma.setX(sma.getX() + (dq[dq.size() - 1].getX() - dq[0].getX()) / mSMAslots);
  sma.setY(sma.getY() + (dq[dq.size() - 1].getY() - dq[0].getY()) / mSMAslots);
  sma.setZ(sma.getZ() + (dq[dq.size() - 1].getZ() - dq[0].getZ()) / mSMAslots);
  sma.setSigmaX(sma.getSigmaX() + (dq[dq.size() - 1].getSigmaX() - dq[0].getSigmaX()) / mSMAslots);
  sma.setSigmaY(sma.getSigmaY() + (dq[dq.size() - 1].getSigmaY() - dq[0].getSigmaY()) / mSMAslots);
  sma.setSigmaZ(sma.getSigmaZ() + (dq[dq.size() - 1].getSigmaZ() - dq[0].getSigmaZ()) / mSMAslots);

  dq.pop_front();

  return;
}

//_____________________________________________
Slot& MeanVertexCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<MeanVertexData>(mUseFit, mNBinsX, mRangeX, mNBinsY, mRangeY, mNBinsZ, mRangeZ));
  return slot;
}

} // end namespace calibration
} // end namespace o2
