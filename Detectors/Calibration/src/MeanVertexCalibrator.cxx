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

#include "Framework/Logger.h"
#include "DetectorsCalibration/MeanVertexCalibrator.h"
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
using CovMatrix = ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>>;

void MeanVertexCalibrator::initOutput()
{
  // Here we initialize the vector of our output objects
  mMeanVertexVector.clear();
  mInfoVector.clear();
  return;
}

//_____________________________________________
void MeanVertexCalibrator::printVector(float* vect, int sizeVect, float minRange, float maxRange)
{
  float binWidth = (maxRange - minRange) / sizeVect;
  for (int i = 0; i < sizeVect; ++i) {
    LOG(info) << "i-th bin [" << minRange + i * binWidth << ", " << minRange + (i + 1) * binWidth << "] = " << i << ", content of histogram = " << vect[i];
  }
  LOG(info) << "Printing to be used to fill a ROOT histogram";
  for (int i = 0; i < sizeVect; ++i) {
    if (vect[i] != 0) {
      LOG(info) << "h->SetBinContent(" << i + 1 << ", " << vect[i] << ");";
    }
  }
}

//_____________________________________________
void MeanVertexCalibrator::printVector(std::vector<float>& vect, float minRange, float maxRange)
{
  printVector(&vect[0], vect.size(), minRange, maxRange);
}

//_____________________________________________
void MeanVertexCalibrator::binVector(std::vector<float>& vectOut, const std::vector<float>& vectIn, int nbins, float min, float max)
{
  vectOut.clear();
  vectOut.resize(nbins);
  float binWidthInv = nbins / (max - min);
  for (int i = 0; i < vectIn.size(); ++i) {
    if (vectIn[i] < min) {
      continue;
    }
    int bin = (vectIn[i] - min) * binWidthInv;
    vectOut[bin]++;
  }
}

//_____________________________________________
void MeanVertexCalibrator::fitMeanVertex(o2::calibration::MeanVertexData* c, MeanVertexObject& mvo)
{
  // x
  fitMeanVertexCoord(0, c->nbinsX, &c->histoX[0], -(c->rangeX), c->rangeX, mvo);

  // y
  fitMeanVertexCoord(1, c->nbinsY, &c->histoY[0], -(c->rangeY), c->rangeY, mvo);

  // z
  fitMeanVertexCoord(2, c->nbinsZ, &c->histoZ[0], -(c->rangeZ), c->rangeZ, mvo);

  // now we do the fits in slices of Z
  // we fit as soon as we have enough entries in z
  double fitres;
  // first we order the vector
  std::sort(c->histoVtx.begin(), c->histoVtx.end(), [](std::array<float, 3> a, std::array<float, 3> b) { return b[2] > a[2]; });
  if (mVerbose) {
    LOG(info) << "Printing ordered vertices";
    for (int i = 0; i < c->histoVtx.size(); ++i) {
      LOG(info) << "x = " << c->histoVtx[i][0] << ", y = " << c->histoVtx[i][1] << ", z = " << c->histoVtx[i][2];
    }
  }

  std::vector<float> htmpX;
  std::vector<float> htmpY;
  std::vector<std::array<double, 3>> fitResSlicesX;
  std::vector<CovMatrix> covMatrixX;
  std::vector<std::array<double, 3>> fitResSlicesY;
  std::vector<CovMatrix> covMatrixY;
  std::vector<double> meanZvect;
  int startZ = 0;
  int counter = 0;
  while (startZ < c->histoVtx.size()) {
    double meanZ = 0;
    int counts = 0;
    for (int ii = startZ; ii < c->histoVtx.size(); ++ii) {
      if (htmpX.size() < mMinEntries) {
        htmpX.push_back(c->histoVtx[ii][0]);
        htmpY.push_back(c->histoVtx[ii][1]);
        meanZ += c->histoVtx[ii][2];
        ++counts;
      } else {
        // we can fit and restart filling
        // X:
        fitResSlicesX.push_back({});
        covMatrixX.push_back({});
        if (mVerbose) {
          LOG(info) << "Fitting X for counter " << counter << ", will use " << c->nbinsX << " bins, from " << -(c->rangeX) << " to " << c->rangeX;
          for (int i = 0; i < htmpX.size(); ++i) {
            LOG(info) << i << " : " << htmpX[i];
          }
        }
        std::vector<float> binnedVect;
        binVector(binnedVect, htmpX, c->nbinsX, -(c->rangeX), c->rangeX);
        if (mVerbose) {
          LOG(info) << " Printing output binned vector for X:";
          printVector(binnedVect, -(c->rangeX), c->rangeX);
        }
        fitres = fitGaus(c->nbinsX, &binnedVect[0], -(c->rangeX), c->rangeX, fitResSlicesX.back(), &covMatrixX.back());
        if (fitres >= 0) {
          LOG(info) << "X, counter " << counter << ": Fit result (of single Slot, z slice [" << c->histoVtx[startZ][2] << ", " << c->histoVtx[ii][2] << "[) => " << fitres << ". Mean = " << fitResSlicesX[counter][1] << " Sigma = " << fitResSlicesX[counter][2] << ", covMatrix = " << covMatrixX[counter](2, 2);
        } else {
          LOG(error) << "X, counter " << counter << ": Fit failed with result = " << fitres;
        }
        htmpX.clear();
        // Y:
        fitResSlicesY.push_back({});
        covMatrixY.push_back({});
        if (mVerbose) {
          LOG(info) << "Fitting Y for counter " << counter << ", will use " << c->nbinsY << " bins, from " << -(c->rangeY) << " to " << c->rangeY;
          for (int i = 0; i < htmpY.size(); ++i) {
            LOG(info) << i << " : " << htmpY[i];
          }
        }
        binnedVect.clear();
        binVector(binnedVect, htmpY, c->nbinsY, -(c->rangeY), c->rangeY);
        if (mVerbose) {
          LOG(info) << " Printing output binned vector for Y:";
          printVector(binnedVect, -(c->rangeY), c->rangeY);
        }
        fitres = fitGaus(c->nbinsY, &binnedVect[0], -(c->rangeY), c->rangeY, fitResSlicesY.back(), &covMatrixY.back());
        if (fitres >= 0) {
          LOG(info) << "Y, counter " << counter << ": Fit result (of single Slot, z slice [" << c->histoVtx[startZ][2] << ", " << c->histoVtx[ii][2] << "[) => " << fitres << ". Mean = " << fitResSlicesY[counter][1] << " Sigma = " << fitResSlicesY[counter][2] << ", covMatrix = " << covMatrixY[counter](2, 2);
        } else {
          LOG(error) << "Y, counter " << counter << ": Fit failed with result = " << fitres;
        }
        htmpY.clear();
        if (mVerbose) {
          LOG(info) << "Z, counter " << counter << ": " << meanZ / counts;
        }
        ++counter;
        meanZvect.push_back(meanZ / counts);
        break;
      }
    }
    startZ += mMinEntries * counter;
  }

  // now we update the error on x
  float sumX, sumY = 0;
  for (int iFit = 0; iFit < counter; ++iFit) {
    sumX += fitResSlicesX[iFit][2] * fitResSlicesX[iFit][2] * covMatrixX[iFit](2, 2) * covMatrixX[iFit](2, 2);
    sumY += fitResSlicesY[iFit][2] * fitResSlicesY[iFit][2] * covMatrixY[iFit](2, 2) * covMatrixY[iFit](2, 2);
  }
  if (mVerbose) {
    LOG(info) << "sumX = " << sumX;
    LOG(info) << "sumY = " << sumY;
  }
  float sigmaX = 1. / std::sqrt(sumX);
  float sigmaY = 1. / std::sqrt(sumY);
  if (mVerbose) {
    LOG(info) << "sigmaX = " << sigmaX;
    LOG(info) << "sigmaY = " << sigmaY;
  }
  mvo.setSigmaX(sigmaX);
  mvo.setSigmaY(sigmaY);

  // now we get the slope for the x-coordinate dependence on z
  TLinearFitter lf(1, "pol1");
  lf.StoreData(kFALSE);
  for (int i = 0; i < fitResSlicesX.size(); ++i) {
    if (mVerbose) {
      LOG(info) << "Adding point " << i << ": zvtx = " << meanZvect[i] << " xvtx = " << fitResSlicesX[i][2];
    }
    lf.AddPoint(&meanZvect[i], fitResSlicesX[i][1]);
  }
  lf.Eval();
  double slopeX = lf.GetParameter(0);
  lf.ClearPoints();

  // now slope for the y-coordinate dependence on z
  for (int i = 0; i < fitResSlicesX.size(); ++i) {
    if (mVerbose) {
      LOG(info) << "Adding point " << i << ": zvtx = " << meanZvect[i] << " yvtx = " << fitResSlicesY[i][2];
    }
    lf.AddPoint(&meanZvect[i], fitResSlicesY[i][1]);
  }
  lf.Eval();
  double slopeY = lf.GetParameter(0);
  if (mVerbose) {
    LOG(info) << "slope X = " << slopeX;
    LOG(info) << "slope Y = " << slopeY;
  }
  mvo.setSlopeX(slopeX);
  mvo.setSlopeY(slopeY);
}
//_____________________________________________
void MeanVertexCalibrator::fitMeanVertexCoord(int icoord, int nbins, float* array, float minRange, float maxRange, MeanVertexObject& mvo)
{
  // fit mean vertex coordinate icoord
  std::vector<float> fitValues;
  if (mVerbose) {
    LOG(info) << "**** Printing content of MeanVertex object for coordinate " << icoord;
    printVector(array, nbins, minRange, maxRange);
  }
  double fitres = fitGaus(nbins, array, minRange, maxRange, fitValues);
  if (fitres >= 0) {
    LOG(info) << "coordinate " << icoord << ": Fit result (of single Slot) => " << fitres << ". Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
  } else {
    LOG(error) << "coordinate " << icoord << ": Fit failed with result = " << fitres;
  }
  mvo.set(icoord, fitValues[1]);
  mvo.setSigma(icoord, fitValues[2]);
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
    fitMeanVertex(c, mvo);
    // now we add the object to the deque
    mTmpMVobjDq.push_back(std::move(mvo));
  } else {
    // we merge the input from the different slots
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
    fitMeanVertex(&mSMAdata, mSMAMVobj);
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
    sma.setSlopeX((sma.getSlopeX() * (dq.size() - 1) + dq.back().getSlopeX()) / dq.size());
    sma.setSlopeY((sma.getSlopeY() * (dq.size() - 1) + dq.back().getSlopeY()) / dq.size());
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
  sma.setSlopeX(sma.getSlopeX() + (dq[dq.size() - 1].getSlopeX() - dq[0].getSlopeX()) / mSMAslots);
  sma.setSlopeY(sma.getSlopeY() + (dq[dq.size() - 1].getSlopeY() - dq[0].getSlopeY()) / mSMAslots);

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
