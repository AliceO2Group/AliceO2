// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <TGeoManager.h>
#include <TRandom.h>

#include "FairLogger.h"
#include "DetectorsBase/GeometryManager.h"

#include "TRDBase/TRDGeometry.h"
#include "TRDBase/TRDSimParam.h"
#include "TRDBase/TRDPadPlane.h"
#include "TRDBase/PadResponse.h"

#include "TRDSimulation/Digitizer.h"
#include "TRDSimulation/TRDSimParams.h"
#include <cmath>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::trd;
using namespace o2::math_utils;

Digitizer::Digitizer()
{
  o2::base::GeometryManager::loadGeometry();
  mGeo = TRDGeometry::instance();
  mGeo->createClusterMatrixArray();          // Requiered for chamberInGeometry()
  mPRF = new PadResponse();                  // Pad response function initialization
  mSimParam = TRDSimParam::Instance();       // Instance for simulation parameters
  mCommonParam = TRDCommonParam::Instance(); // Instance for common parameters
  if (!mSimParam) {
  }
  if (!mCommonParam) {
  } else {
    if (!mCommonParam->cacheMagField()) {
    }
  }

  // obtain the number of threads from configuration
#ifdef WITH_OPENMP
  int askedthreads = TRDSimParams::Instance().digithreads;
  int maxthreads = omp_get_max_threads();
  if (askedthreads < 0) {
    mNumThreads = maxthreads;
  } else {
    mNumThreads = std::min(maxthreads, askedthreads);
  }
  LOG(INFO) << "TRD: Digitizing with " << mNumThreads << " threads ";
#endif

  // initialize structures that we need per thread
  mGausRandomRings.resize(mNumThreads);
  mFlatRandomRings.resize(mNumThreads);
  mLogRandomRings.resize(mNumThreads);
  for (int i = 0; i < mNumThreads; ++i) {
    mGausRandomRings[i].initialize(RandomRing<>::RandomType::Gaus);
    mFlatRandomRings[i].initialize(RandomRing<>::RandomType::Flat);
    mLogRandomRings[i].initialize([]() -> float { return std::log(gRandom->Rndm()); });
    mDriftEstimators.emplace_back();
  }

  mSDigits = false;
}

void Digitizer::process(std::vector<HitType> const& hits, DigitContainer& digitCont, o2::dataformats::MCTruthContainer<MCLabel>& labels)
{
  if (!mCalib) {
    LOG(FATAL) << "TRD Calibration database not available";
  }

  // TODO: it might be worth making these member variables
  // in order to have less memory allocations
  std::array<SignalContainer, kNdet> signalsMapCollection;
  std::array<DigitContainer, kNdet> digitCollection;
  std::array<o2::dataformats::MCTruthContainer<MCLabel>, kNdet> labelsperdetector;

  // Get the a hit container for all the hits in a given detector then call convertHits for a given detector (0 - 539)
  std::array<std::vector<HitType>, kNdet> hitsPerDetector;
  getHitContainerPerDetector(hits, hitsPerDetector);

#ifdef WITH_OPENMP
  omp_set_num_threads(mNumThreads);
// Loop over all TRD detectors (in a parallel fashion)
#pragma omp parallel for schedule(dynamic)
#endif
  for (int det = 0; det < kNdet; ++det) {
#ifdef WITH_OPENMP
    const int threadid = omp_get_thread_num();
#else
    const int threadid = 0;
#endif
    auto& signalsMap = signalsMapCollection[det];
    auto& digits = digitCollection[det];
    // Jump to the next detector if the detector is
    // switched off, not installed, etc
    if (mCalib->isChamberNoData(det)) {
      continue;
    }
    if (!mGeo->chamberInGeometry(det)) {
      continue;
    }

    // Go to the next detector if there are no hits
    if (hitsPerDetector[det].size() == 0) {
      continue;
    }

    if (!convertHits(det, hitsPerDetector[det], signalsMap, labelsperdetector[det], threadid)) {
      LOG(WARN) << "TRD conversion of hits failed for detector " << det;
      continue; // go to the next chamber
    }

    // O2-790
    if (signalsMap.size() == 0) {
      continue; // go to the next chamber
    }

    if (!convertSignalsToADC(det, signalsMap, digits, threadid)) {
      LOG(WARN) << "TRD conversion of signals to digits failed for detector " << det;
      continue; // go to the next chamber
    }
  }

  // Finalize: Dump the digitCollection to the output digitCont
  for (int det = 0; det < kNdet; ++det) {
    auto& digits = digitCollection[det];
    // digitCont.insert(digitCont.end(), digits.begin(), digits.end());
    std::move(digits.begin(), digits.end(), std::back_inserter(digitCont));
    // digitCont.insert(digitCont.end(), std::make_move_iterator(digits.begin()), std::make_move_iterator(digits.end()));
    labels.mergeAtBack(labelsperdetector[det]);
  }
}

void Digitizer::getHitContainerPerDetector(const std::vector<HitType>& hits, std::array<std::vector<HitType>, kNdet>& hitsPerDetector)
{
  //
  // Fill an array of size kNdet (540)
  // The i-element of the array contains the hit collection for the i-detector
  // To be called once, before doing the loop over all detectors and process the hits
  //
  for (const auto& hit : hits) {
    hitsPerDetector[hit.GetDetectorID()].push_back(hit);
  }
}

bool Digitizer::convertHits(const int det, const std::vector<HitType>& hits, SignalContainer& signalMapCont, o2::dataformats::MCTruthContainer<MCLabel>& labels, int thread)
{
  //
  // Convert the detector-wise sorted hits to detector signals
  //
  const int kNpad = mSimParam->getNumberOfPadsInPadResponse(); // Number of pads included in the pad response
  const float kAmWidth = TRDGeometry::amThick();               // Width of the amplification region
  const float kDrWidth = TRDGeometry::drThick();               // Width of the drift retion
  const float kDrMin = -0.5 * kAmWidth;                        // Drift + Amplification region
  const float kDrMax = kDrWidth + 0.5 * kAmWidth;              // Drift + Amplification region

  int timeBinTRFend = 0;
  double padSignal[kNpad];

  if (mSimParam->TRFOn()) {
    timeBinTRFend = ((int)(mSimParam->GetTRFhi() * mCommonParam->GetSamplingFrequency())) - 1;
  }

  const double calExBDetValue = mCalib->getExB(det); // T * V/cm (check units)
  const int nTimeTotal = kTimeBins;                  // PLEASE FIX ME when CCDB is ready
  const float samplingRate = mCommonParam->GetSamplingFrequency();
  const float elAttachProp = mSimParam->GetElAttachProp() / 100;

  const TRDPadPlane* padPlane = mGeo->getPadPlane(det);
  const int layer = mGeo->getLayer(det);
  const float rowEndROC = padPlane->getRowEndROC();
  const float row0 = padPlane->getRow0ROC();
  const int nRowMax = padPlane->getNrows();
  const int nColMax = padPlane->getNcols();

  // Loop over hits
  for (const auto& hit : hits) {
    bool isDigit = false;
    size_t labelIndex = labels.getIndexedSize();
    const int qTotal = hit.GetCharge();
    /*
      Now the real local coordinate system of the ROC
      column direction: locC
      row direction:    locR
      time direction:   locT
      locR and locC are identical to the coordinates of the corresponding
      volumina of the drift or amplification region.
      locT is defined relative to the wire plane (i.e. middle of amplification
      region), meaning locT = 0, and is negative for hits coming from the
      drift region.
    */
    double locC = hit.getLocalC();        // col direction in amplification or drift volume
    double locR = hit.getLocalR();        // row direction in amplification or drift volume
    double locT = hit.getLocalT();        // time direction in amplification or drift volume
    const double driftLength = -1 * locT; // The drift length in cm without diffusion
    // Patch to take care of TR photons that are absorbed
    // outside the chamber volume. A real fix would actually need
    // a more clever implementation of the TR hit generation
    if (qTotal < 0) {
      if ((locR < rowEndROC) || (locR > row0)) {
        continue;
      }
      if ((driftLength < kDrMin) || (driftLength > kDrMax)) {
        continue;
      }
    }

    int rowE = padPlane->getPadRowNumberROC(locR);
    if (rowE < 0) {
      continue;
    }

    double rowOffset = padPlane->getPadRowOffsetROC(rowE, locR);
    double offsetTilt = padPlane->getTiltOffset(rowOffset);
    int colE = padPlane->getPadColNumber(locC + offsetTilt);
    if (colE < 0) {
      continue;
    }

    double absDriftLength = std::fabs(driftLength); // Normalized drift length
    if (mCommonParam->ExBOn()) {
      absDriftLength /= std::sqrt(1 / (1 + calExBDetValue * calExBDetValue));
    }

    float driftVelocity = mCalib->getVDrift(det, colE, rowE); // The drift velocity
    float t0 = mCalib->getT0(det, colE, rowE);                // The T0 velocity

    // Loop over all created electrons
    const int nElectrons = std::fabs(qTotal);
    for (int el = 0; el < nElectrons; ++el) {
      // Electron attachment
      if (mSimParam->ElAttachOn()) {
        if (mFlatRandomRings[thread].getNextValue() < absDriftLength * elAttachProp) {
          continue;
        }
      }
      // scoped diffused coordinates for each electron
      double locRd{locR}, locCd{locC}, locTd{locT};

      // Apply diffusion smearing
      if (mSimParam->DiffusionOn()) {
        if (!diffusion(driftVelocity, absDriftLength, calExBDetValue, locR, locC, locT, locRd, locCd, locTd, thread)) {
          continue;
        }
      }

      // Apply E x B effects
      if (mCommonParam->ExBOn()) {
        locCd = locCd + calExBDetValue * driftLength;
      }
      // The electron position after diffusion and ExB in pad coordinates.
      rowE = padPlane->getPadRowNumberROC(locRd);
      if (rowE < 0) {
        continue;
      }
      rowOffset = padPlane->getPadRowOffsetROC(rowE, locRd);
      // The pad column (rphi-direction)
      offsetTilt = padPlane->getTiltOffset(rowOffset);
      colE = padPlane->getPadColNumber(locCd + offsetTilt);
      if (colE < 0) {
        continue;
      }
      const double colOffset = padPlane->getPadColOffset(colE, locCd + offsetTilt);
      driftVelocity = mCalib->getVDrift(det, colE, rowE); // The drift velocity for the updated col and row
      t0 = mCalib->getT0(det, colE, rowE);                // The T0 velocity for the updated col and row
      // Convert the position to drift time [mus], using either constant drift velocity or
      // time structure of drift cells (non-isochronity, GARFIELD calculation).
      // Also add absolute time of hits to take pile-up events into account properly
      double driftTime;
      if (mSimParam->TimeStructOn()) {
        // Get z-position with respect to anode wire
        double zz = row0 - locR + padPlane->getAnodeWireOffset();
        zz -= ((int)(2 * zz)) * 0.5;
        if (zz > 0.25) {
          zz = 0.5 - zz;
        }
        // Use drift time map (GARFIELD)
        driftTime = mDriftEstimators[thread].TimeStruct(driftVelocity, 0.5 * kAmWidth - 1.0 * locTd, zz) + hit.GetTime();
      } else {
        // Use constant drift velocity
        driftTime = std::fabs(locTd) / driftVelocity + hit.GetTime();
      }

      // Apply the gas gain including fluctuations
      const double signal = -(mSimParam->GetGasGain()) * mLogRandomRings[thread].getNextValue();

      // Apply the pad response
      if (mSimParam->PRFOn()) {
        // The distance of the electron to the center of the pad in units of pad width
        double dist = (colOffset - 0.5 * padPlane->getColSize(colE)) / padPlane->getColSize(colE);
        // ********************************************************************************
        // This is a fixed parametrization, i.e. not dependent on calibration values !
        // ********************************************************************************
        if (!(mPRF->getPRF(signal, dist, layer, padSignal))) {
          continue;
        }
      } else {
        padSignal[0] = 0;
        padSignal[1] = signal;
        padSignal[2] = 0;
      }
      // The time bin (always positive), with t0 distortion
      double timeBinIdeal = driftTime * samplingRate + t0;
      // Protection
      if (std::fabs(timeBinIdeal) > (2 * nTimeTotal)) {
        timeBinIdeal = 2 * nTimeTotal;
      }
      int timeBinTruncated = ((int)timeBinIdeal);
      // The distance of the position to the middle of the timebin
      double timeOffset = ((float)timeBinTruncated + 0.5 - timeBinIdeal) / samplingRate;
      // Sample the time response inside the drift region + additional time bins before and after.
      // The sampling is done always in the middle of the time bin
      const int firstTimeBin = std::max(timeBinTruncated, 0);
      const int lastTimeBin = std::min(timeBinTruncated + timeBinTRFend, nTimeTotal);
      // loop over pads first then over timebins for better cache friendliness
      // and less access to signalMapCont
      for (int pad = 0; pad < kNpad; ++pad) {
        int colPos = colE + pad - 1;
        if (colPos < 0) {
          continue;
        }
        if (colPos >= nColMax) {
          break;
        }
        const int key = calculateKey(det, rowE, colPos);
        if (key < KEY_MIN || key > KEY_MAX) {
          LOG(FATAL) << "Wrong TRD key " << key << " for (det,row,col) = (" << det << ", " << rowE << ", " << colPos << ")";
        }
        isDigit = true;
        auto& currentSignalData = signalMapCont[key]; // Get the old signal or make a new one if it doesn't exist
        auto& currentSignal = currentSignalData.signals;
        currentSignalData.labelIndex = labelIndex;
        for (int tb = firstTimeBin; tb < lastTimeBin; ++tb) {
          // Apply the time response
          double timeResponse = 1;
          double crossTalk = 0;
          const double t = (tb - timeBinTruncated) / samplingRate + timeOffset;
          if (mSimParam->TRFOn()) {
            timeResponse = mSimParam->TimeResponse(t);
          }
          if (mSimParam->CTOn()) {
            crossTalk = mSimParam->CrossTalk(t);
          }
          float signalOld = currentSignal[tb];
          if (colPos != colE) {
            // Cross talk added to non-central pads
            signalOld += padSignal[pad] * (timeResponse + crossTalk);
          } else {
            // Without cross talk at central pad
            signalOld += padSignal[pad] * timeResponse;
          }
          // Update the final signal
          currentSignal[tb] = signalOld;
        } // Loop: time bins
      }   // Loop: pads
    }     // end of loop over electrons
    if (isDigit) {
      MCLabel label(hit.GetTrackID(), getEventID(), getSrcID()); // add one label if at least one digit is created
      labels.addElement(labelIndex, label);
    }
  } // end of loop over hits
  return true;
}

float drawGaus(o2::math_utils::RandomRing<>& normaldistRing, float mu, float sigma)
{
  // this is using standard normally distributed random numbers and rescaling to make
  // them gaussian distributed with general mu and sigma
  return mu + sigma * normaldistRing.getNextValue();
}

bool Digitizer::convertSignalsToADC(const int det, SignalContainer& signalMapCont, DigitContainer& digits, int thread)
{
  //
  // Converts the sampled electron signals to ADC values for a given chamber
  //
  if (signalMapCont.size() == 0) {
    return false;
  }

  constexpr double kEl2fC = 1.602e-19 * 1.0e15;                                 // Converts number of electrons to fC
  double coupling = mSimParam->GetPadCoupling() * mSimParam->GetTimeCoupling(); // Coupling factor
  double convert = kEl2fC * mSimParam->GetChipGain();                           // Electronics conversion factor
  double adcConvert = mSimParam->GetADCoutRange() / mSimParam->GetADCinRange(); // ADC conversion factor
  double baseline = mSimParam->GetADCbaseline() / adcConvert;                   // The electronics baseline in mV
  double baselineEl = baseline / convert;                                       // The electronics baseline in electrons

  int nTimeTotal = kTimeBins; // fDigitsManager->GetDigitsParam()->GetNTimeBins(det);

  for (auto& signalMapIter : signalMapCont) {
    const auto key = signalMapIter.first;
    const int det = getDetectorFromKey(key);
    const int row = getRowFromKey(key);
    const int col = getColFromKey(key);
    // halfchamber masking
    int mcm = (int)(col / 18);               // current group of 18 col pads
    int halfchamberside = (mcm > 3 ? 1 : 0); // 0=Aside, 1=Bside

    // Halfchambers that are switched off, masked by mCalib
    /* Something is wrong with isHalfChamberNoData - deactivated for now
    if (mCalib->isHalfChamberNoData(det, halfchamberside)) {
      continue;
    }
    */

    // Check whether pad is masked
    // Bridged pads are not considered yet!!!
    if (mCalib->isPadMasked(det, col, row) || mCalib->isPadNotConnected(det, col, row)) {
      continue;
    }

    float padgain = mCalib->getPadGainFactor(det, row, col); // The gain factor
    if (padgain <= 0) {
      LOG(FATAL) << "Not a valid gain " << padgain << ", " << det << ", " << col << ", " << row;
    }

    // Loop over the all timebins in the ADC array
    SignalArray& signalData = signalMapIter.second;
    auto& signalArray = signalData.signals;
    ArrayADC adcs{};
    for (int tb = 0; tb < nTimeTotal; ++tb) {
      float signalAmp = (float)signalArray[tb]; // The signal amplitude
      signalAmp *= coupling;                    // Pad and time coupling
      signalAmp *= padgain;                     // Gain factors
      // Add the noise, starting from minus ADC baseline in electrons
      signalAmp = std::max((double)drawGaus(mGausRandomRings[thread], signalAmp, mSimParam->GetNoise()), -baselineEl);
      signalAmp *= convert;  // Convert to mV
      signalAmp += baseline; // Add ADC baseline in mV
      // Convert to ADC counts
      // Set the overflow-bit fADCoutRange if the signal is larger than fADCinRange
      ADC_t adc = 0;
      if (signalAmp >= mSimParam->GetADCinRange()) {
        adc = ((ADC_t)mSimParam->GetADCoutRange());
      } else {
        adc = std::lround(signalAmp * adcConvert);
      }
      // update the adc array value
      adcs[tb] = adc;
    } // loop over timebins
    // Convert the map to digits here, and push them to the container
    size_t labelIndex = signalData.labelIndex;
    digits.emplace_back(det, row, col, adcs, labelIndex, getEventTime());
  } // loop over digits
  return true;
}

bool Digitizer::diffusion(float vdrift, float absdriftlength, float exbvalue,
                          float lRow0, float lCol0, float lTime0,
                          double& lRow, double& lCol, double& lTime, int thread)
{
  //
  // Applies the diffusion smearing to the position of a single electron.
  // Depends on absolute drift length.
  //
  float diffL = 0.0;
  float diffT = 0.0;
  if (mDriftEstimators[thread].GetDiffCoeff(diffL, diffT, vdrift)) {
    float driftSqrt = std::sqrt(absdriftlength);
    float sigmaT = driftSqrt * diffT;
    float sigmaL = driftSqrt * diffL;
    lRow = drawGaus(mGausRandomRings[thread], lRow0, sigmaT);
    if (mCommonParam->ExBOn()) {
      const float exbfactor = 1.f / (1.f + exbvalue * exbvalue);
      lCol = drawGaus(mGausRandomRings[thread], lCol0, sigmaT * exbfactor);
      lTime = drawGaus(mGausRandomRings[thread], lTime0, sigmaL * exbfactor);
    } else {
      lCol = drawGaus(mGausRandomRings[thread], lCol0, sigmaT);
      lTime = drawGaus(mGausRandomRings[thread], lTime0, sigmaL);
    }
    return true;
  } else {
    return false;
  }
}
