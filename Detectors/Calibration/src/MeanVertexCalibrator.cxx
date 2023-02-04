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
#include "DetectorsCalibration/MeanVertexParams.h"

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
void MeanVertexCalibrator::printVector(const float* vect, const MeanVertexCalibrator::HistoParams& hpar)
{
  for (int i = 0; i < hpar.nBins; ++i) {
    LOG(info) << "i-th bin [" << hpar.minRange + i * hpar.binWidth << ", " << hpar.minRange + (i + 1) * hpar.binWidth << "] = " << i << ", content of histogram = " << vect[i];
  }
  LOG(info) << "Printing to be used as a vector holding the content";
  for (int i = 0; i < hpar.nBins; ++i) {
    LOG(info) << "vect[" << i << "] = " << vect[i] << ";";
  }
  LOG(info) << "Printing to be used to fill a ROOT histogram";
  for (int i = 0; i < hpar.nBins; ++i) {
    if (vect[i] != 0) {
      LOG(info) << "h->SetBinContent(" << i + 1 << ", " << vect[i] << ");";
    }
  }
}

//_____________________________________________
void MeanVertexCalibrator::printVector(const std::vector<float>& vect, const MeanVertexCalibrator::HistoParams& hpar)
{
  printVector(vect.data(), hpar);
}

//_____________________________________________
MeanVertexCalibrator::HistoParams MeanVertexCalibrator::binVector(std::vector<float>& vectOut, const std::vector<float>& vectIn, o2::calibration::MeanVertexData* c, int dim)
{
  const char* dimName[3] = {"X", "Y", "Z"};
  vectOut.clear();
  // define binning
  const auto& params = MeanVertexParams::Instance();
  float mean = c->getMean(dim), rms = c->getRMS(dim);
  if (rms < params.minSigma[dim]) {
    LOGP(alarm, "Too small RMS = {} for dimension {} ({} entries), override to {}", rms, dimName[dim], c->entries, params.minSigma[dim]);
    rms = params.minSigma[dim];
  }
  float minD = mean - params.histoNSigma[dim] * rms;
  float maxD = mean + params.histoNSigma[dim] * rms;
  int nBins = std::max(1.f, (maxD - minD) / params.histoBinSize[dim]);
  float binWidth = (maxD - minD) / nBins, binWidthInv = 1. / binWidth;
  if (mVerbose) {
    LOGP(info, "Histo for dim:{} with {} entries: mean:{} rms:{} -> {} bins in range {}:{} ", dimName[dim], c->entries, mean, rms, nBins, minD, maxD, nBins);
  }
  vectOut.resize(nBins);
  for (int i = 0; i < vectIn.size(); ++i) {
    if (vectIn[i] < minD) {
      continue;
    }
    int bin = (vectIn[i] - minD) * binWidthInv;
    if (bin >= nBins) {
      continue;
    }
    vectOut[bin]++;
  }
  return {nBins, binWidth, minD, maxD};
}

//_____________________________________________
bool MeanVertexCalibrator::fitMeanVertex(o2::calibration::MeanVertexData* c, MeanVertexObject& mvo)
{
  // now we do the fits in slices of Z
  // we fit as soon as we have enough entries in z
  const auto& params = MeanVertexParams::Instance();
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
  int startZ = 0, nBinsOK = 0;
  auto minEntriesPerPoint = std::max((unsigned long int)params.minEntries, c->histoVtx.size() / params.nPointsForSlope);
  if (mVerbose) {
    LOGP(info, "Beginning: startZ = {} c->histoVtx.size() = {}, will process {} Z slices with {} entries each", startZ, c->histoVtx.size(), params.nPointsForSlope, minEntriesPerPoint);
  }
  auto dumpNonEmpty = [&binnedVect](const std::string& msg) {
    if (!msg.empty()) {
      LOG(info) << msg;
    }
    for (size_t i = 0; i < binnedVect.size(); i++) {
      if (binnedVect[i]) {
        LOGP(info, "bin:{} {}", i, binnedVect[i]);
      }
    }
  };

  for (int counter = 0; counter < params.nPointsForSlope; counter++) {
    if (mVerbose) {
      LOG(info) << "Beginning of while: startZ = " << startZ << " c->histoVtx.size() = " << c->histoVtx.size();
    }
    double meanZ = 0;
    int counts = 0;
    for (int ii = startZ; ii < c->histoVtx.size(); ++ii) {
      bool failed = false;
      if (++counts < minEntriesPerPoint) {
        htmpX.push_back(c->histoVtx[ii][0]);
        htmpY.push_back(c->histoVtx[ii][1]);
        meanZ += c->histoVtx[ii][2];
      } else {
        counts--;
        if (mVerbose) {
          LOGP(info, "fitting after collecting {} entries for Z slice {} of {}", htmpX.size(), counter, params.nPointsForSlope);
        }
        // we can fit and restart filling
        // X:
        fitResSlicesX.push_back({});
        covMatrixX.push_back({});
        auto hparX = binVector(binnedVect, htmpX, c, 0);
        if (mVerbose) {
          LOG(info) << "Fitting X for counter " << counter << ", will use " << hparX.nBins << " bins, from " << hparX.minRange << " to " << hparX.maxRange;
          for (int i = 0; i < htmpX.size(); ++i) {
            LOG(info) << "vect[" << i << "] = " << htmpX[i] << ";";
          }
        }
        if (mVerbose) {
          LOG(info) << " Printing output binned vector for X:";
          printVector(binnedVect, hparX);
        } else if (params.dumpNonEmptyBins) {
          dumpNonEmpty(fmt::format("X{} nonEmpty bins", counter));
        }
        fitres = fitGaus(hparX.nBins, binnedVect.data(), hparX.minRange, hparX.maxRange, fitResSlicesX.back(), &covMatrixX.back());
        if (fitres != -10) {
          LOG(info) << "X, counter " << counter << ": Fit result (z slice [" << c->histoVtx[startZ][2] << ", " << c->histoVtx[ii][2] << "]) => " << fitres << ". Mean = " << fitResSlicesX[counter][1] << " Sigma = " << fitResSlicesX[counter][2] << ", covMatrix = " << covMatrixX[counter](2, 2) << " entries = " << counts;
        } else {
          LOG(error) << "X, counter " << counter << ": Fit failed with result = " << fitres << " entries = " << counts;
          failed = true;
        }
        htmpX.clear();

        // Y:
        fitResSlicesY.push_back({});
        covMatrixY.push_back({});
        binnedVect.clear();
        auto hparY = binVector(binnedVect, htmpY, c, 1);
        if (mVerbose) {
          LOG(info) << "Fitting Y for counter " << counter << ", will use " << hparY.nBins << " bins, from " << hparY.minRange << " to " << hparY.maxRange;
          for (int i = 0; i < htmpY.size(); ++i) {
            LOG(info) << i << " : " << htmpY[i];
          }
        }
        if (mVerbose) {
          LOG(info) << " Printing output binned vector for Y:";
          printVector(binnedVect, hparY);
        } else if (params.dumpNonEmptyBins) {
          dumpNonEmpty(fmt::format("Y{} nonEmpty bins", counter));
        }
        fitres = fitGaus(hparY.nBins, binnedVect.data(), hparY.minRange, hparY.maxRange, fitResSlicesY.back(), &covMatrixY.back());
        if (fitres != -10) {
          LOG(info) << "Y, counter " << counter << ": Fit result (z slice [" << c->histoVtx[startZ][2] << ", " << c->histoVtx[ii][2] << "]) => " << fitres << ". Mean = " << fitResSlicesY[counter][1] << " Sigma = " << fitResSlicesY[counter][2] << ", covMatrix = " << covMatrixY[counter](2, 2) << " entries = " << counts;
        } else {
          LOG(error) << "Y, counter " << counter << ": Fit failed with result = " << fitres << " entries = " << counts;
          failed = true;
        }
        htmpY.clear();

        // Z: let's calculate the mean position
        if (mVerbose) {
          LOGP(info, "Z, counter {} {} ({}/{})", counter, meanZ / counts, meanZ, counts);
        }

        if (failed) {
          fitResSlicesX.pop_back();
          covMatrixX.pop_back();
          fitResSlicesY.pop_back();
          covMatrixY.pop_back();
        } else {
          meanZvect.push_back(meanZ / counts);
          nBinsOK++;
        }
        startZ += counts;
        break;
      }
    }
    if (mVerbose) {
      LOG(info) << "End of while: startZ = " << startZ << " c->histoVtx.size() = " << c->histoVtx.size();
    }
  }

  // fitting main mean vtx Z
  for (int ii = 0; ii < c->histoVtx.size(); ++ii) {
    htmpZ.push_back(c->histoVtx[ii][2]);
  }
  auto hparZ = binVector(binnedVect, htmpZ, c, 2);
  fitMeanVertexCoord(2, binnedVect.data(), hparZ, mvo);
  htmpZ.clear();
  binnedVect.clear();

  // now we update the error on x
  double sumX = 0, sumY = 0, weightSumX = 0, weightSumY = 0;
  for (int iFit = 0; iFit < nBinsOK; ++iFit) {
    if (mVerbose) {
      LOG(info) << "SigmaX = " << fitResSlicesX[iFit][2] << " error = " << covMatrixX[iFit](2, 2);
      LOG(info) << "SigmaY = " << fitResSlicesY[iFit][2] << " error = " << covMatrixY[iFit](2, 2);
    }
    double weightSigmaX = covMatrixX[iFit](2, 2) > 0 ? 1. / covMatrixX[iFit](2, 2) : 1.; // covMatrix is already an error squared
    sumX += (fitResSlicesX[iFit][2] * weightSigmaX);
    weightSumX += weightSigmaX;

    double weightSigmaY = covMatrixY[iFit](2, 2) > 0 ? 1. / covMatrixY[iFit](2, 2) : 1.; // covMatrix is already an error squared
    sumY += (fitResSlicesY[iFit][2] * weightSigmaY);
    weightSumY += weightSigmaY;
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
  if (sigmaX == 0 || sigmaY == 0 || mvo.getSigmaZ() == 0 || nBinsOK < 2) {
    LOGP(error, "Fit with {} valid slices produced wrong vertex parameters: SigmaX={}, SigmaY={}, SigmaZ={}", nBinsOK, sigmaX, sigmaY, mvo.getSigmaZ());
    return false;
  }
  mvo.setSigmaX(sigmaX);
  mvo.setSigmaY(sigmaY);

  // now we get the slope for the x-coordinate dependence on z
  TLinearFitter lf(1, "pol1");
  lf.StoreData(kFALSE);
  for (int i = 0; i < nBinsOK; ++i) {
    if (mVerbose) {
      LOG(info) << "Adding point " << i << ": zvtx = " << meanZvect[i] << " xvtx = " << fitResSlicesX[i][2];
    }
    lf.AddPoint(&meanZvect[i], fitResSlicesX[i][1]);
  }
  lf.Eval();
  double slopeX = lf.GetParameter(1);
  mvo.setSlopeX(slopeX);
  mvo.setX(mvo.getZ() * slopeX + lf.GetParameter(0));
  lf.ClearPoints();

  // now slope for the y-coordinate dependence on z
  for (int i = 0; i < nBinsOK; ++i) {
    if (mVerbose) {
      LOG(info) << "Adding point " << i << ": zvtx = " << meanZvect[i] << " yvtx = " << fitResSlicesY[i][2];
    }
    lf.AddPoint(&meanZvect[i], fitResSlicesY[i][1]);
  }
  lf.Eval();
  double slopeY = lf.GetParameter(1);
  mvo.setSlopeY(slopeY);
  mvo.setY(mvo.getZ() * slopeY + lf.GetParameter(0));
  if (mVerbose) {
    LOG(info) << "slope X = " << slopeX;
    LOG(info) << "slope Y = " << slopeY;
  }
  LOG(info) << "Fitted meanVertex: " << mvo.asString();
  return true;
}
//_____________________________________________
void MeanVertexCalibrator::fitMeanVertexCoord(int icoord, const float* array, const MeanVertexCalibrator::HistoParams& hpar, MeanVertexObject& mvo)
{
  // fit mean vertex coordinate icoord
  std::vector<float> fitValues;
  float binWidth = 0;
  if (mVerbose) {
    LOG(info) << "**** Printing content of MeanVertex object for coordinate " << icoord;
    printVector(array, hpar);
  }
  double fitres = fitGaus(hpar.nBins, array, hpar.minRange, hpar.maxRange, fitValues);
  if (fitres != -4) {
    LOG(info) << "coordinate " << icoord << ": Fit result of full statistics => " << fitres << ". Mean = " << fitValues[1] << " Sigma = " << fitValues[2];
  } else {
    LOG(error) << "coordinate " << icoord << ": Fit failed with result = " << fitres;
  }
  switch (icoord) {
    case 0:
      mvo.setX(fitValues[1]);
      mvo.setSigmaX(fitValues[2]);
      break;
    case 1:
      mvo.setY(fitValues[1]);
      mvo.setSigmaY(fitValues[2]);
      break;
    case 2:
      mvo.setZ(fitValues[1]);
      mvo.setSigmaZ(fitValues[2]);
      break;
  }
}

//_____________________________________________
void MeanVertexCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  const auto& params = MeanVertexParams::Instance();
  o2::calibration::MeanVertexData* c = slot.getContainer();
  LOG(info) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd() << " with "
            << c->getEntries() << " entries";
  MeanVertexObject mvo;
  // fitting
  if (!fitMeanVertex(c, mvo)) {
    return;
  }
  mTmpMVobjDqTime.emplace_back(slot.getStartTimeMS(), slot.getEndTimeMS());
  // now we add the object to the deque
  mTmpMVobjDq.push_back(std::move(mvo));

  // moving average
  doSimpleMovingAverage(mTmpMVobjDq, mSMAMVobj);

  if (mTmpMVobjDqTime.size() > params.nSlots4SMA) {
    mTmpMVobjDqTime.pop_front();
  }
  long offset = (slot.getEndTimeMS() - slot.getStartTimeMS()) / 2;
  long startValidity = (mTmpMVobjDqTime.front().getMin() + mTmpMVobjDqTime.back().getMax()) / 2 - offset;
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
  if (dq.size() <= MeanVertexParams::Instance().nSlots4SMA) {
    sma = std::accumulate(dq.begin(), dq.end(), 0.0) / dq.size();
    //avg = (avg * (vect.size() - 1) + vect.back()) / vect.size();
    return;
  }

  // if the vector has size > mSMAslots, we calculate the SMA, and then we drop 1 element
  // (note that it can have a size > mSMAslots only of 1 element!)
  sma += (dq[dq.size() - 1] - dq[0]) / MeanVertexParams::Instance().nSlots4SMA;
  dq.pop_front();

  return;
}

//_____________________________________________
void MeanVertexCalibrator::doSimpleMovingAverage(std::deque<MVObject>& dq, MVObject& sma)
{

  // doing simple moving average
  const auto& params = MeanVertexParams::Instance();
  if (dq.size() <= params.nSlots4SMA) {
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
  sma.setX(sma.getX() + (dq[dq.size() - 1].getX() - dq[0].getX()) / params.nSlots4SMA);
  sma.setY(sma.getY() + (dq[dq.size() - 1].getY() - dq[0].getY()) / params.nSlots4SMA);
  sma.setZ(sma.getZ() + (dq[dq.size() - 1].getZ() - dq[0].getZ()) / params.nSlots4SMA);
  sma.setSigmaX(sma.getSigmaX() + (dq[dq.size() - 1].getSigmaX() - dq[0].getSigmaX()) / params.nSlots4SMA);
  sma.setSigmaY(sma.getSigmaY() + (dq[dq.size() - 1].getSigmaY() - dq[0].getSigmaY()) / params.nSlots4SMA);
  sma.setSigmaZ(sma.getSigmaZ() + (dq[dq.size() - 1].getSigmaZ() - dq[0].getSigmaZ()) / params.nSlots4SMA);
  sma.setSlopeX(sma.getSlopeX() + (dq[dq.size() - 1].getSlopeX() - dq[0].getSlopeX()) / params.nSlots4SMA);
  sma.setSlopeY(sma.getSlopeY() + (dq[dq.size() - 1].getSlopeY() - dq[0].getSlopeY()) / params.nSlots4SMA);

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
  slot.setContainer(std::make_unique<MeanVertexData>());
  return slot;
}

bool MeanVertexCalibrator::hasEnoughData(const Slot& slot) const
{
  auto minReq = MeanVertexParams::Instance().minEntries * MeanVertexParams::Instance().nPointsForSlope;
  if (mVerbose) {
    LOG(info) << "container * " << (void*)slot.getContainer();
    LOG(info) << "container entries = " << slot.getContainer()->entries << ", minEntries = " << minReq;
  }
  return slot.getContainer()->entries >= minReq;
}

} // end namespace calibration
} // end namespace o2
