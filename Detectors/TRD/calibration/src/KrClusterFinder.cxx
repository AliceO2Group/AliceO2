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

/// \file KrClusterFinder.cxx
/// \brief The TRD Krypton cluster finder from digits
/// \author Alexander Schmah, Ole Schmidt

#include "TRDCalibration/KrClusterFinder.h"
#include "Math/IntegratorOptions.h"
#include <fairlogger/Logger.h>

#include <limits>

using namespace o2::trd;
using namespace o2::trd::constants;

double KrClusterFinder::LandauChi2Functor::operator()(const double* par) const
{
  // provides chi2 estimate comparing y[] to par[0] * TMath::Landau(x[], par[1], par[2])
  // par[0] : amplitude
  // par[1] : location parameter (approximately most probable value)
  // par[2] : sigma
  double retVal = 0;
  for (int i = xLowerBound; i <= xUpperBound; ++i) {
    if (fabs(y[i]) < 1e-3f) {
      // exclude bins with zero errors like in TH1::Fit
      // the standard bin error is the square root of its content
      continue;
    }
    retVal += TMath::Power(y[i] - par[0] * TMath::Landau(x[i], par[1], par[2]), 2) / y[i];
  }
  return retVal;
}

double KrClusterFinder::getRms(const std::vector<uint64_t>& adcIndices, int itTrunc, double nRmsTrunc, int minAdc, double& rmsTime, uint32_t& sumAdc) const
{
  double rmsAdc = 1e7;
  double meanAdc = -10.;
  rmsTime = 1e7;
  double meanTime = -10.f;

  for (int it = 0; it < itTrunc; ++it) {
    // iterations for truncated mean
    sumAdc = 0;
    uint32_t sumAdc2 = 0;
    uint32_t sumTime = 0;
    uint32_t sumTime2 = 0;
    uint32_t sumWeights = 0;
    for (const auto& adcIdx : adcIndices) {
      int iTb = adcIdx % TIMEBINS;
      int iDigit = adcIdx / TIMEBINS;
      int adc = mDigits[iDigit].getADC()[iTb] - mBaselineAdc;
      if (adc < minAdc) {
        continue;
      }
      if (fabs(adc - meanAdc) < nRmsTrunc * rmsAdc) {
        sumAdc += adc;
        sumAdc2 += adc * adc;
        sumWeights += 1;
        sumTime += iTb;
        sumTime2 += iTb * iTb;
      }
    }
    if (sumWeights > 0) {
      auto sumWeightsInv = 1. / sumWeights;
      meanAdc = sumAdc * sumWeightsInv;
      rmsAdc = TMath::Sqrt(sumWeightsInv * (sumAdc2 - 2. * meanAdc * sumAdc + sumWeights * meanAdc * meanAdc));
      meanTime = sumTime * sumWeightsInv;
      rmsTime = TMath::Sqrt(sumWeightsInv * (sumTime2 - 2. * meanTime * sumTime + sumWeights * meanTime * meanTime));
    }
  }
  return rmsAdc;
}

void KrClusterFinder::init()
{
  mFitter.SetFCN<LandauChi2Functor>(3, mLandauChi2Functor, mInitialFitParams.data());
  mFitter.Config().ParSettings(0).SetLimits(0., 1.e5);
  mFitter.Config().ParSettings(1).SetLimits(0., 30.);
  mFitter.Config().ParSettings(2).SetLimits(1.e-3, 20.);
  mFuncLandauFit = std::make_unique<TF1>(
    "fLandauFit", [&](double* x, double* par) { return par[0] * TMath::Landau(x[0], par[1], par[2]); }, 0., static_cast<double>(TIMEBINS), 3);
}

void KrClusterFinder::reset()
{
  mKrClusters.clear();
  mTrigRecs.clear();
}

void KrClusterFinder::setInput(const gsl::span<const Digit>& digitsIn, const gsl::span<const TriggerRecord>& trigRecIn)
{
  mDigits = digitsIn;
  mTriggerRecords = trigRecIn;
}

void KrClusterFinder::findClusters()
{
  if (mDigits.size() == 0 || mTriggerRecords.size() == 0) {
    return;
  }
  int nClsTotal = 0;
  int nClsDropped = 0;
  int nClsInvalidFit = 0;
  /*
  The input digits and trigger records are provided on a per time frame basis.
  The cluster finding is performed for each trigger. In order to not copy the
  input data we sort the digits for each trigger by their detector ID and keep
  the sorted indices in the digitIdxArray.
  */
  std::vector<uint64_t> digitIdxArray(mDigits.size());
  std::iota(digitIdxArray.begin(), digitIdxArray.end(), 0);
  for (const auto& trig : mTriggerRecords) {
    // sort the digits of given trigger record by detector ID
    const auto& digits = mDigits; // this reference is only needed to be able to pass it to the lambda for sorting
    std::stable_sort(std::begin(digitIdxArray) + trig.getFirstDigit(), std::begin(digitIdxArray) + trig.getNumberOfDigits() + trig.getFirstDigit(),
                     [&digits](uint64_t i, uint64_t j) { return digits[i].getDetector() < digits[j].getDetector(); });

    /*
    Count number of digits per detector:
    For each detector we need to know the number of digits.
    We fill the array idxFirstDigitInDet with the total number of digits
    up to given detector. So we start at 0 for detector 0 and end up with
    the total number of digits in entry 540. Thus the number of digits in
    detector iDet can be calculated with:
    idxFirstDigitInDet[iDet+1] - idxFirstDigitInDet[iDet]
    Note, that for a global index one needs to add trig.getFirstDigit()
    */
    int currDet = 0;
    int nextDet = 0;
    int digitCounter = 0;
    std::array<unsigned int, MAXCHAMBER + 1> idxFirstDigitInDet{};
    auto idxHelperPtr = &(idxFirstDigitInDet.data()[1]);
    idxHelperPtr[-1] = 0;
    for (int iDigit = trig.getFirstDigit(); iDigit < trig.getFirstDigit() + trig.getNumberOfDigits(); ++iDigit) {
      if (mDigits[digitIdxArray[iDigit]].getDetector() > currDet) {
        nextDet = mDigits[digitIdxArray[iDigit]].getDetector();
        for (int iDet = currDet; iDet < nextDet; ++iDet) {
          idxHelperPtr[iDet] = digitCounter;
        }
        currDet = nextDet;
      }
      ++digitCounter;
    }
    for (int iDet = currDet; iDet <= MAXCHAMBER; ++iDet) {
      idxHelperPtr[iDet] = digitCounter;
    }

    // the cluster search is done on a per detector basis
    for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
      unsigned int nDigitsInDet = idxFirstDigitInDet[iDet + 1] - idxFirstDigitInDet[iDet];
      std::vector<bool> isAdcUsed(nDigitsInDet * TIMEBINS); // keep track of the ADC values which have already been checked or added to a cluster
      std::vector<bool> isDigitUsed(nDigitsInDet);          // keep track of the digits which have already been processed
      bool continueClusterSearch = true;
      // the following loop searches for one cluster at a time in iDet
      while (continueClusterSearch) {

        // start by finding the max ADC value in all digits of iDet
        uint16_t adcMax = 0;
        int tbMax = 0;
        int rowMax = 0;
        int colMax = 0;
        for (unsigned int iDigit = 0; iDigit < nDigitsInDet; ++iDigit) {
          uint64_t digitIdx = digitIdxArray[trig.getFirstDigit() + idxFirstDigitInDet[iDet] + iDigit]; // global index for array of all digits (mDigits)
          if (mDigits[digitIdx].isSharedDigit()) {
            // we need to skip the shared digits which are duplicates contained in the global digits array
            continue;
          }
          if (isDigitUsed[iDigit]) {
            // if a maximum has been found for this digit then all ADCs above threshold are already flagged as used
            continue;
          }
          int tbMaxADC = -1;
          auto maxAdcInDigit = mDigits[digitIdx].getADCmax(tbMaxADC);
          if (maxAdcInDigit > adcMax) {
            // potentially found a maximum
            tbMax = tbMaxADC;
            rowMax = mDigits[digitIdx].getPadRow();
            colMax = mDigits[digitIdx].getPadCol();
            adcMax = maxAdcInDigit;
          }
        }
        if (adcMax < mMinAdcForMax) {
          // the maximum ADC value is below the threshold, go to next chamber
          break;
        }

        // cluster around max ADC value
        int lowerTb = TIMEBINS;
        int upperTb = 0;
        int lowerCol = NCOLUMN;
        int upperCol = 0;
        int lowerRow = NROWC1;
        int upperRow = 0;
        int nUsedADCsInCl = 0;
        std::vector<uint64_t> constituentAdcIndices;
        for (unsigned int iDigit = 0; iDigit < nDigitsInDet; ++iDigit) {
          uint64_t digitIdx = digitIdxArray[trig.getFirstDigit() + idxFirstDigitInDet[iDet] + iDigit]; // global index for array of all digits (mDigits)
          if (mDigits[digitIdx].isSharedDigit()) {
            // we need to skip the shared digits which are duplicates contained in the global digits array
            continue;
          }
          int row = mDigits[digitIdx].getPadRow();
          if (std::abs(row - rowMax) > 1) {
            continue;
          }
          int col = mDigits[digitIdx].getPadCol();
          if (std::abs(col - colMax) > 2) {
            continue;
          }
          bool addedAdc = false;
          for (int iTb = 0; iTb < TIMEBINS; ++iTb) {
            if (isAdcUsed[iDigit * TIMEBINS + iTb]) {
              continue;
            }
            // flag this ADC value as used, regardless of whether or not it is added to the cluster
            // (if it is below the threshold it won't be used for another cluster anyway)
            isAdcUsed[iDigit * TIMEBINS + iTb] = true;
            if (mDigits[digitIdx].getADC()[iTb] > mMinAdcClContrib) {
              addedAdc = true;
              if (iTb < lowerTb) {
                lowerTb = iTb;
              }
              if (iTb > upperTb) {
                upperTb = iTb;
              }
              ++nUsedADCsInCl;
            }
            constituentAdcIndices.push_back(digitIdx * TIMEBINS + iTb); // also add ADC values below threshold here
          }
          if (addedAdc) {
            isDigitUsed[iDigit] = true;
            if (row < lowerRow) {
              lowerRow = row;
            }
            if (row > upperRow) {
              upperRow = row;
            }
            if (col < lowerCol) {
              lowerCol = col;
            }
            if (col > upperCol) {
              upperCol = col;
            }
          }
        }

        // determine cluster size
        int clSizeTime = upperTb - lowerTb;
        int clSizeRow = upperRow - lowerRow;
        int clSizeCol = upperCol - lowerCol;

        if (nUsedADCsInCl > 0) {
          // after this cluster is processed, continue looking for another one in iDet
          continueClusterSearch = true;
        }

        // sum up ADC values (total sum, sum per time bin, total sum for ADCs above energy threshold)
        std::array<int, TIMEBINS> sumPerTb{};
        int sumOfAllTimeBins = 0;
        int sumOfAllTimeBinsAboveThreshold = 0;
        for (const auto idx : constituentAdcIndices) {
          auto iTb = idx % TIMEBINS;
          auto iDigit = idx / TIMEBINS;
          sumOfAllTimeBins += mDigits[iDigit].getADC()[iTb] - mBaselineAdc;
          sumPerTb[iTb] += mDigits[iDigit].getADC()[iTb] - mBaselineAdc;
          if (mDigits[iDigit].getADC()[iTb] > mMinAdcClEoverT) {
            sumOfAllTimeBinsAboveThreshold += mDigits[iDigit].getADC()[iTb] - mBaselineAdc;
          }
        }

        uint32_t sumOfAdcTrunc;
        double rmsTimeTrunc;
        auto rmsAdcClusterTrunc = getRms(constituentAdcIndices, 2, 3., static_cast<uint32_t>(mMinAdcClEoverT * .95), rmsTimeTrunc, sumOfAdcTrunc);
        (void)rmsAdcClusterTrunc; // return value not used, so silence compiler warning about unused variable

        // ADC value and time bin of first maximum
        int maxAdcA = -1;
        int maxTbA = -1;
        mLandauChi2Functor.x.clear();
        mLandauChi2Functor.y.clear();
        for (int iTb = 0; iTb < TIMEBINS; ++iTb) {
          mLandauChi2Functor.x.push_back((float)iTb);
          mLandauChi2Functor.y.push_back(sumPerTb[iTb]);
          if (sumPerTb[iTb] < mMinAdcForMax) {
            continue;
          }
          if (sumPerTb[iTb] > maxAdcA) {
            maxAdcA = sumPerTb[iTb];
            maxTbA = iTb;
          }
        }
        mLandauChi2Functor.xLowerBound = maxAdcA - 1;
        mLandauChi2Functor.xUpperBound = maxAdcA + 2;

        // ADC value and time bin of second maximum (if there is one)
        int maxAdcB = -1;
        int maxTbB = -1;
        for (int iTb = 2; iTb < TIMEBINS - 2; ++iTb) { // we need to check the neighbouring two bins, so ignore the outermost two bins in the search for a second maximum
          if (std::abs(maxTbA - iTb) < 3 || sumPerTb[iTb] < mMinAdcForSecondMax) {
            // we are too close to the first maximum (or the ADC value is not too small)
            continue;
          }
          if (sumPerTb[iTb] > maxAdcB) {
            // check neighbours
            if (sumPerTb[iTb - 1] < sumPerTb[iTb] &&
                sumPerTb[iTb - 2] < sumPerTb[iTb] &&
                sumPerTb[iTb + 1] < sumPerTb[iTb] &&
                sumPerTb[iTb + 2] < sumPerTb[iTb]) {
              maxAdcB = sumPerTb[iTb];
              maxTbB = iTb;
            }
          }
        }

        // quality checks for maxima
        bool isBadA = false;
        bool isBadB = false;
        if (maxAdcA < 0 || maxTbA <= 1 || maxTbA >= (TIMEBINS - 2)) {
          isBadA = true;
        }
        if (maxAdcB < 0) {
          // second maximum must be inside the time acceptance by construction
          isBadB = true;
        }
        if (!isBadA) {
          // we have a good first maximum, let's check its shape
          if ((sumPerTb[maxTbA - 1] / static_cast<double>(maxAdcA)) < 0.1 || (sumPerTb[maxTbA + 1] / static_cast<double>(maxAdcA)) < 0.25) {
            isBadA = true;
          }
        }
        if (!isBadB) {
          // we have a good second maximum, let's check its shape
          if ((sumPerTb[maxTbB - 1] / static_cast<double>(maxAdcB)) < 0.1 || (sumPerTb[maxTbB + 1] / static_cast<double>(maxAdcB)) < 0.25) {
            isBadB = true;
          }
        }
        if (!isBadA && !isBadB) {
          // we have two maxima, check order and size
          if (maxTbA > maxTbB || maxAdcA <= maxAdcB) {
            isBadA = true;
            isBadB = true;
          }
        }

        if (clSizeCol > 0 && clSizeTime > 3 && clSizeTime < TIMEBINS) {
          mFitter.Config().ParSettings(0).SetValue(maxAdcA);
          mFitter.Config().ParSettings(1).SetValue(maxTbA);
          mFitter.Config().ParSettings(2).SetValue(.5);
          bool fitOK = mFitter.FitFCN();
          if (!fitOK) {
            ++nClsInvalidFit;
          }
          mFuncLandauFit->SetParameters(mFitResult->GetParams()[0], mFitResult->GetParams()[1], mFitResult->GetParams()[2]);
          double integralLandauFit = fitOK ? mFuncLandauFit->Integral(0., static_cast<double>(TIMEBINS), 1.e-9) : 0.;

          double rmsTime;
          uint32_t rmsSumAdc;
          auto rmsAdc = getRms(constituentAdcIndices, 1, 2.5, mMinAdcClContrib, rmsTime, rmsSumAdc);

          double sumAdcA = 0.;
          double sumAdcB = 0.;
          if (!isBadA && !isBadB) {
            // use the Landau fit to the first peak to extrapolate the energy of larger time bins (close time bins are evaluated from the array directly)
            // for the second peak the Landau fit is used to subtract the energy from the first peak
            for (int iTb = 0; iTb < TIMEBINS; ++iTb) {
              if (iTb < (maxTbB - 2)) {
                sumAdcA += mLandauChi2Functor.y[iTb];
              } else {
                sumAdcA += mFuncLandauFit->Eval(mLandauChi2Functor.x[iTb]);
                sumAdcB += mLandauChi2Functor.y[iTb] - mFuncLandauFit->Eval(mLandauChi2Functor.x[iTb]);
              }
            }
          }
          // create Kr cluster
          if (sumOfAllTimeBins <= std::numeric_limits<uint16_t>::max() &&
              sumOfAllTimeBinsAboveThreshold <= std::numeric_limits<uint16_t>::max() &&
              integralLandauFit <= std::numeric_limits<uint16_t>::max() &&
              sumAdcA <= std::numeric_limits<uint16_t>::max() &&
              sumAdcB <= std::numeric_limits<uint16_t>::max() &&
              rmsAdc <= std::numeric_limits<uint16_t>::max() &&
              rmsTime <= std::numeric_limits<uint8_t>::max() &&
              nUsedADCsInCl <= std::numeric_limits<uint8_t>::max() &&
              sumOfAdcTrunc <= std::numeric_limits<uint16_t>::max()) {
            if (maxTbA < 0) {
              maxTbA = tbMax;
            }
            if (maxTbB < 0) {
              maxTbB = 2 * TIMEBINS + 5; // larger than any maximum time bin we can have
            }
            KrCluster cluster;
            cluster.setGlobalPadID(iDet, rowMax, colMax);
            cluster.setAdcData(sumOfAllTimeBins, (int)rmsAdc, (int)sumAdcA, (int)sumAdcB, sumOfAllTimeBinsAboveThreshold, (int)integralLandauFit, sumOfAdcTrunc);
            cluster.setTimeData(maxTbA, maxTbB, (int)rmsTime);
            cluster.setClusterSizeData(clSizeRow, clSizeCol, clSizeTime, nUsedADCsInCl);
            mKrClusters.push_back(cluster);
            ++nClsTotal;
          } else {
            //mFitResult->Print(std::cout);
            ++nClsDropped;
            LOG(debug) << "Kr cluster cannot be added because values are out of range";
            LOGF(debug, "sumOfAllTimeBins(%i), sumAdcA(%f), sumAdcB(%f), clSizeRow(%i), clSizeCol(%i), clSizeTime(%i), maxTbA(%i), maxTbB(%i)", sumOfAllTimeBins, sumAdcA, sumAdcB, clSizeRow, clSizeCol, clSizeTime, maxTbA, maxTbB);
            LOGF(debug, "rmsAdc(%f), rmsTime(%f), nUsedADCsInCl(%i), sumOfAllTimeBinsAboveThreshold(%i), integralLandauFit(%f), sumOfAdcTrunc(%u)", rmsAdc, rmsTime, nUsedADCsInCl, sumOfAllTimeBinsAboveThreshold, integralLandauFit, sumOfAdcTrunc);
          }
        }
      } // end cluster search
    }   // end detector loop
  }     // end trigger loop

  // we don't need the exact BC time, just use first interaction record within this TF
  mTrigRecs.emplace_back(mTriggerRecords[0].getBCData(), nClsTotal);
  LOGF(info, "Number of Kr clusters with a) invalid fit (%i) b) out-of-range values which were dropped (%i)", nClsInvalidFit, nClsDropped);
}
