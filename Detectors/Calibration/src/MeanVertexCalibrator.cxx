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
void MeanVertexCalibrator::printVector(float* vect, int sizeVect, float minRange, float maxRange, float binWidth)
{
  for (int i = 0; i < sizeVect; ++i) {
    LOG(info) << "i-th bin [" << minRange + i * binWidth << ", " << minRange + (i + 1) * binWidth << "] = " << i << ", content of histogram = " << vect[i];
  }
  LOG(info) << "Printing to be used as a vector holding the content";
  for (int i = 0; i < sizeVect; ++i) {
    LOG(info) << "vect[" << i << "] = " << vect[i] << ";";
  }
  LOG(info) << "Printing to be used to fill a ROOT histogram";
  for (int i = 0; i < sizeVect; ++i) {
    if (vect[i] != 0) {
      LOG(info) << "h->SetBinContent(" << i + 1 << ", " << vect[i] << ");";
    }
  }
}

//_____________________________________________
void MeanVertexCalibrator::printVector(std::vector<float>& vect, float minRange, float maxRange, float binWidth)
{
  printVector(&vect[0], vect.size(), minRange, maxRange, binWidth);
}

//_____________________________________________
void MeanVertexCalibrator::binVector(std::vector<float>& vectOut, const std::vector<float>& vectIn, int nbins, float min, float max, float binWidthInv)
{
  vectOut.clear();
  vectOut.resize(nbins);
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
  std::vector<float> htmpZ;
  std::vector<std::array<double, 3>> fitResSlicesX;
  std::vector<CovMatrix> covMatrixX;
  std::vector<std::array<double, 3>> fitResSlicesY;
  std::vector<CovMatrix> covMatrixY;
  std::vector<float> binnedVect;
  std::vector<double> meanZvect;
  int startZ = 0;
  int counter = 0;
  int minEntriesPerPoint = std::max(uint64_t(mMinEntries), c->histoVtx.size() / mNPointsForSlope);
  if (mVerbose) {
    LOG(info) << "Beginning: startZ = " << startZ << " c->histoVtx.size() = " << c->histoVtx.size();
  }
  while (startZ <= c->histoVtx.size()) {
    if (mVerbose) {
      LOG(info) << "Beginning of while: startZ = " << startZ << " c->histoVtx.size() = " << c->histoVtx.size();
    }
    double meanZ = 0;
    int counts = 0;
    for (int ii = startZ; ii <= c->histoVtx.size(); ++ii) {
      if (mVerbose) {
        // LOG(info) << "htmpX.size() = " << htmpX.size() << " ii = " << ii << " c->histoVtx.size() = " << c->histoVtx.size();
      }
      if (htmpX.size() < minEntriesPerPoint) {
        if (mVerbose) {
          // LOG(info) << "filling X with c->histoVtx[" << ii << "][0] = " << c->histoVtx[ii][0];
          // LOG(info) << "filling Y with c->histoVtx[" << ii << "][0] = " << c->histoVtx[ii][1];
        }
        htmpX.push_back(c->histoVtx[ii][0]);
        htmpY.push_back(c->histoVtx[ii][1]);
        meanZ += c->histoVtx[ii][2];
        ++counts;
      } else {
        if (mVerbose) {
          LOG(info) << "fitting ";
        }
        // we can fit and restart filling
        // X:
        fitResSlicesX.push_back({});
        covMatrixX.push_back({});
        if (mVerbose) {
          LOG(info) << "Fitting X for counter " << counter << ", will use " << mNBinsX << " bins, from " << -mRangeX << " to " << mRangeX;
          for (int i = 0; i < htmpX.size(); ++i) {
            LOG(info) << "vect[" << i << "] = " << htmpX[i] << ";";
          }
        }
        binVector(binnedVect, htmpX, mNBinsX, -mRangeX, mRangeX, mBinWidthXInv);
        if (mVerbose) {
          LOG(info) << " Printing output binned vector for X:";
          printVector(binnedVect, -(mRangeX), mRangeX, mBinWidthX);
        }
        fitres = fitGaus(mNBinsX, &binnedVect[0], -(mRangeX), mRangeX, fitResSlicesX.back(), &covMatrixX.back());
        if (fitres != 10) {
          LOG(info) << "X, counter " << counter << ": Fit result (z slice [" << c->histoVtx[startZ][2] << ", " << c->histoVtx[ii][2] << "[) => " << fitres << ". Mean = " << fitResSlicesX[counter][1] << " Sigma = " << fitResSlicesX[counter][2] << ", covMatrix = " << covMatrixX[counter](2, 2);
        } else {
          LOG(error) << "X, counter " << counter << ": Fit failed with result = " << fitres;
        }
        htmpX.clear();

        // Y:
        fitResSlicesY.push_back({});
        covMatrixY.push_back({});
        if (mVerbose) {
          LOG(info) << "Fitting Y for counter " << counter << ", will use " << mNBinsY << " bins, from " << -(mRangeY) << " to " << mRangeY;
          for (int i = 0; i < htmpY.size(); ++i) {
            LOG(info) << i << " : " << htmpY[i];
          }
        }
        binnedVect.clear();
        binVector(binnedVect, htmpY, mNBinsY, -(mRangeY), mRangeY, mBinWidthYInv);
        if (mVerbose) {
          LOG(info) << " Printing output binned vector for Y:";
          printVector(binnedVect, -(mRangeY), mRangeY, mBinWidthY);
        }
        fitres = fitGaus(mNBinsY, &binnedVect[0], -(mRangeY), mRangeY, fitResSlicesY.back(), &covMatrixY.back());
        if (fitres != 10) {
          LOG(info) << "Y, counter " << counter << ": Fit result (z slice [" << c->histoVtx[startZ][2] << ", " << c->histoVtx[ii][2] << "[) => " << fitres << ". Mean = " << fitResSlicesY[counter][1] << " Sigma = " << fitResSlicesY[counter][2] << ", covMatrix = " << covMatrixY[counter](2, 2);
        } else {
          LOG(error) << "Y, counter " << counter << ": Fit failed with result = " << fitres;
        }
        htmpY.clear();

        // Z: let's calculate the mean position
        if (mVerbose) {
          LOG(info) << "Z, counter " << counter << ": " << meanZ / counts;
        }
        ++counter;
        meanZvect.push_back(meanZ / counts);
        break;
      }
    }
    startZ += mMinEntries * counter;
    if (mVerbose) {
      LOG(info) << "End of while: startZ = " << startZ << " c->histoVtx.size() = " << c->histoVtx.size();
    }
  }

  // fitting main mean vtx Z
  for (int ii = 0; ii < c->histoVtx.size(); ++ii) {
    htmpZ.push_back(c->histoVtx[ii][2]);
  }
  binVector(binnedVect, htmpZ, mNBinsZ, -(mRangeZ), mRangeZ, mBinWidthZInv);
  fitMeanVertexCoord(2, mNBinsZ, &binnedVect[0], -(mRangeZ), mRangeZ, mvo);
  htmpZ.clear();
  binnedVect.clear();

  // now we update the error on x
  double sumX = 0, sumY = 0, weightSumX = 0, weightSumY = 0;
  for (int iFit = 0; iFit < counter; ++iFit) {
    if (mVerbose) {
      LOG(info) << "SigmaX = " << fitResSlicesX[iFit][2] << " error = " << covMatrixX[iFit](2, 2);
      LOG(info) << "SigmaY = " << fitResSlicesY[iFit][2] << " error = " << covMatrixY[iFit](2, 2);
    }
    if (covMatrixX[iFit](2, 2) != 0) {
      double weightSigma = 1. / covMatrixX[iFit](2, 2); // covMatrix is already an error squared
      sumX += (fitResSlicesX[iFit][2] * weightSigma);
      weightSumX += weightSigma;
    }
    if (covMatrixY[iFit](2, 2) != 0) {
      double weightSigma = 1. / covMatrixY[iFit](2, 2); // covMatrix is already an error squared
      sumY += (fitResSlicesY[iFit][2] * weightSigma);
      weightSumY += weightSigma;
    }
  }
  if (mVerbose) {
    LOG(info) << "sumX = " << sumX;
    LOG(info) << "weightSumX = " << weightSumX;
    LOG(info) << "sumY = " << sumY;
    LOG(info) << "weightSumY = " << weightSumY;
  }

  double sigmaX = 0;
  if (weightSumX != 0) {
    sigmaX = sumX / weightSumX;
  }
  double sigmaY = 0;
  if (weightSumY != 0) {
    sigmaY = sumY / weightSumY;
  }
  if (mVerbose) {
    LOG(info) << "SigmaX for MeanVertex = " << sigmaX;
    LOG(info) << "SigmaY for MeanVertex = " << sigmaY;
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
  mvo.setSlopeX(slopeX);
  mvo.setX(mvo.getZ() * slopeX + lf.GetParameter(1));
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
  mvo.setSlopeY(slopeY);
  mvo.setY(mvo.getZ() * slopeY + lf.GetParameter(1));
  if (mVerbose) {
    LOG(info) << "slope X = " << slopeX;
    LOG(info) << "slope Y = " << slopeY;
  }
}
//_____________________________________________
void MeanVertexCalibrator::fitMeanVertexCoord(int icoord, int nbins, float* array, float minRange, float maxRange, MeanVertexObject& mvo)
{
  // fit mean vertex coordinate icoord
  std::vector<float> fitValues;
  float binWidth = 0;
  if (mVerbose) {
    LOG(info) << "**** Printing content of MeanVertex object for coordinate " << icoord;
    if (icoord == 0) {
      binWidth = mBinWidthX;
    } else if (icoord == 1) {
      binWidth = mBinWidthY;
    } else {
      binWidth = mBinWidthZ;
    }
    printVector(array, nbins, minRange, maxRange, binWidth);
  }
  double fitres = fitGaus(nbins, array, minRange, maxRange, fitValues);
  if (fitres != -4) {
    LOG(info) << "coordinate " << icoord << ": Fit result of full statistics => " << fitres << ". Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
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
  MeanVertexObject mvo;
  // fitting
  fitMeanVertex(c, mvo);
  // now we add the object to the deque
  mTmpMVobjDq.push_back(std::move(mvo));

  // moving average
  doSimpleMovingAverage(mTmpMVobjDq, mSMAMVobj);

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
  if (mVerbose) {
    LOG(info) << "Printing MeanVertex Object:";
    mSMAMVobj.print();
  }

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
    if (mVerbose) {
      LOG(info) << "Printing from simple moving average, when we have not collected enough objects yet:";
      sma.print();
    }
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

  if (mVerbose) {
    LOG(info) << "Printing from simple moving average:";
    sma.print();
  }

  return;
}

//_____________________________________________
Slot& MeanVertexCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  // slot.setContainer(std::make_unique<MeanVertexData>(mNBinsX, mRangeX, mNBinsY, mRangeY, mNBinsZ, mRangeZ));
  slot.setContainer(std::make_unique<MeanVertexData>());
  return slot;
}

} // end namespace calibration
} // end namespace o2
