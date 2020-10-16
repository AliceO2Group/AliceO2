// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsCalibration/MeanVertexCalibrator.h"
#include "Framework/Logger.h"
#include "MathUtils/MathBase.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"

namespace o2
{
namespace calibration
{

using Slot = o2::calibration::TimeSlot<o2::calibration::MeanVertexData>;
using o2::math_utils::math_base::fitGaus;
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
  LOG(INFO) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd() << " with "
            << c->getEntries() << " entries";
  mTmpMVobjDqTimeStart.push_back(slot.getTFStart());
  if (mUseFit) {
    // x coordinate
    std::vector<float> fitValues;
    float* array = &c->histoX[0];
    double fitres = fitGaus(c->nbinsX, array, -(c->rangeX), c->rangeX, fitValues);
    if (fitres >= 0) {
      LOG(INFO) << "X: Fit result (of single Slot) => " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(ERROR) << "X: Fit failed with result = " << fitres;
    }
    mTmpMVobjDqX.push_back(fitValues[1]);
    mTmpMVobjDqSigmaX.push_back(fitValues[2]);

    // y coordinate
    array = &c->histoY[0];
    fitres = fitGaus(c->nbinsY, array, -(c->rangeY), c->rangeY, fitValues);
    if (fitres >= 0) {
      LOG(INFO) << "Y: Fit result (of single Slot) => " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(ERROR) << "Y: Fit failed with result = " << fitres;
    }
    mTmpMVobjDqY.push_back(fitValues[1]);
    mTmpMVobjDqSigmaY.push_back(fitValues[2]);

    // z coordinate
    array = &c->histoZ[0];
    fitres = fitGaus(c->nbinsZ, array, -(c->rangeZ), c->rangeZ, fitValues);
    if (fitres >= 0) {
      LOG(INFO) << "Z: Fit result (of single Slot) => " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(ERROR) << "Z: Fit failed with result = " << fitres;
    }
    mTmpMVobjDqZ.push_back(fitValues[1]);
    mTmpMVobjDqSigmaZ.push_back(fitValues[2]);

  }
  else {
    mTmpMVdataDq.push_back(std::move(*c));
    mSMAdata.merge(&mTmpMVdataDq.back());
    if (mTmpMVobjDqTimeStart.size() > mSMAslots) {
      mSMAdata.subtract(&mTmpMVdataDq.front());
      mTmpMVdataDq.pop_front();
    }
  }

  // output object
  MeanVertexObject mvo;
  
  // now we need to check if we can do some moving averages
  TFType startValidity = std::accumulate(mTmpMVobjDqTimeStart.begin(), mTmpMVobjDqTimeStart.end(), 0.0) / mTmpMVobjDqTimeStart.size();
  if (mUseFit) {
    
    doSimpleMovingAverage(mTmpMVobjDqX, mSMAx);
    doSimpleMovingAverage(mTmpMVobjDqY, mSMAy);
    doSimpleMovingAverage(mTmpMVobjDqZ, mSMAz);
    doSimpleMovingAverage(mTmpMVobjDqSigmaX, mSMAsigmax);
    doSimpleMovingAverage(mTmpMVobjDqSigmaY, mSMAsigmay);
    doSimpleMovingAverage(mTmpMVobjDqSigmaZ, mSMAsigmaz);
    
    mvo.setX(mSMAx);
    mvo.setSigmaX(mSMAsigmax);
    mvo.setY(mSMAy);
    mvo.setSigmaY(mSMAsigmay);
    mvo.setZ(mSMAz);
    mvo.setSigmaZ(mSMAsigmaz);

  }  
  else {
    // now we need to fit, on the merged data
    LOG(DEBUG) << "**** Printing content of SMA MVData object for x coordinate";
    for (int i = 0 ; i < mSMAdata.nbinsX; i++) {
      LOG(DEBUG) << "i = " << i << ", content of histogram = " << mSMAdata.histoX[i];
    }
    std::vector<float> fitValues;
    float* array = &mSMAdata.histoX[0];
    double fitres = fitGaus(mSMAdata.nbinsX, array, -(mSMAdata.rangeX), mSMAdata.rangeX, fitValues);
    if (fitres >= 0) {
      LOG(INFO) << "X: Fit result (of merged Slots) => " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(ERROR) << "X: Fit failed with result = " << fitres;
    }
    mvo.setX(fitValues[1]);
    mvo.setSigmaX(fitValues[2]);

    // y coordinate
    array = &mSMAdata.histoY[0];
    fitres = fitGaus(mSMAdata.nbinsY, array, -(mSMAdata.rangeY), mSMAdata.rangeY, fitValues);
    if (fitres >= 0) {
      LOG(INFO) << "Y: Fit result (of merged Slots) => " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(ERROR) << "Y: Fit failed with result = " << fitres;
    }
    mvo.setY(fitValues[1]);
    mvo.setSigmaY(fitValues[2]);

    // z coordinate
    array = &mSMAdata.histoZ[0];
    fitres = fitGaus(mSMAdata.nbinsZ, array, -(mSMAdata.rangeZ), mSMAdata.rangeZ, fitValues);
    if (fitres >= 0) {
      LOG(INFO) << "Z: Fit result (of merged Slots) => " << fitres << " Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
    } else {
      LOG(ERROR) << "Z: Fit failed with result = " << fitres;
    }
    mvo.setZ(fitValues[1]);
    mvo.setSigmaZ(fitValues[2]);
  }
  
  // TODO: the timestamp is now given with the TF index, but it will have
  // to become an absolute time. This is true both for the lhc phase object itself
  // and the CCDB entry
  std::map<std::string, std::string> md;
  auto clName = o2::utils::MemFileHelper::getClassName(mvo);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  mInfoVector.emplace_back("GRP/MeanVertex", clName, flName, md, startValidity, 99999999999999);
  mMeanVertexVector.emplace_back(mvo);

  slot.print();
}

//_____________________________________________
void MeanVertexCalibrator::doSimpleMovingAverage(std::deque<float>& dq, float& sma) {

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
Slot& MeanVertexCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<MeanVertexData>(mUseFit, mNBinsX, mRangeX, mNBinsY, mRangeY, mNBinsZ, mRangeZ));
  return slot;
}

} // end namespace calibration
} // end namespace o2
