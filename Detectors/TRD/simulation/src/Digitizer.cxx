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

#include "TRDSimulation/Digitizer.h"

#include <TGeoManager.h>
#include <TRandom.h>

#include <fairlogger/Logger.h>
#include "DetectorsBase/GeometryManager.h"

#include "DataFormatsTRD/Hit.h"

#include "TRDBase/Geometry.h"
#include "TRDSimulation/SimParam.h"
#include "TRDBase/PadPlane.h"
#include "TRDBase/PadResponse.h"
#include "TRDSimulation/TRDSimParams.h"

#include <cmath>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace o2::trd;
using namespace o2::trd::constants;
using namespace o2::math_utils;

// init method for late initialization
void Digitizer::init()
{
  mGeo = Geometry::instance();
  mGeo->createClusterMatrixArray();          // Requiered for chamberInGeometry()

  mSimParam.cacheMagField();

  // obtain the number of threads from configuration
#ifdef WITH_OPENMP
  int askedthreads = TRDSimParams::Instance().digithreads;
  int maxthreads = omp_get_max_threads();
  if (askedthreads < 0) {
    mNumThreads = maxthreads;
  } else {
    mNumThreads = std::min(maxthreads, askedthreads);
  }
  LOG(info) << "TRD: Digitizing with " << mNumThreads << " threads ";
#endif

  // initialize structures that we need per thread
  mGausRandomRings.resize(mNumThreads);
  mFlatRandomRings.resize(mNumThreads);
  mLogRandomRings.resize(mNumThreads);
  for (int i = 0; i < mNumThreads; ++i) {
    mGausRandomRings[i].initialize(RandomRing<>::RandomType::Gaus);
    mFlatRandomRings[i].initialize(RandomRing<>::RandomType::Flat);
    mLogRandomRings[i].initialize([]() -> float { return std::log(gRandom->Rndm()); });
    mDriftEstimators.emplace_back(mSimParam.getGasMixture(), mSimParam.getCachedField());
  }

  setSimulationParameters();
}

void Digitizer::setSimulationParameters()
{
  mNpad = mSimParam.getNumberOfPadsInPadResponse(); // Number of pads included in the pad response
  if (mSimParam.trfOn()) {
    mTimeBinTRFend = ((int)(mSimParam.getTRFhi() * mSimParam.getSamplingFrequency())) - 1;
  }
  mMaxTimeBins = TIMEBINS;     // for signals, usually set at 30 tb = 3 microseconds
  mMaxTimeBinsTRAP = TIMEBINS; // for adcs; should be read from the CCDB or the TRAP config
  mSamplingRate = mSimParam.getSamplingFrequency();
  mElAttachProp = mSimParam.getElAttachProp() / 100;
}

void Digitizer::flush(DigitContainer& digits, o2::dataformats::MCTruthContainer<MCLabel>& labels)
{
  if (mPileupSignals.size() > 0) {
    // Add the signals, all chambers are keept in the same signal container
    SignalContainer smc = addSignalsFromPileup();
    if (smc.size() > 0) {
      bool status = convertSignalsToADC(smc, digits);
      if (!status) {
        LOG(warn) << "TRD conversion of signals to digits failed";
      }
      dumpLabels(smc, labels);
    }
  } else {
    // since we don't have any pileup signals just flush the signals for each chamber
    // we avoid flattening the array<map, ndets> to a single map
    for (auto& smc : mSignalsMapCollection) {
      bool status = convertSignalsToADC(smc, digits);
      if (!status) {
        LOG(warn) << "TRD conversion of signals to digits failed";
      }
      dumpLabels(smc, labels);
    }
  }
  clearContainers();
}

void Digitizer::dumpLabels(const SignalContainer& smc, o2::dataformats::MCTruthContainer<MCLabel>& labels)
{
  for (const auto& iter : smc) {
    if (iter.second.isDigit) {
      labels.addElements(labels.getIndexedSize(), iter.second.labels);
      if (iter.second.isShared) {
        labels.addElements(labels.getIndexedSize(), iter.second.labels); // shared digit is a copy of the previous one, need to add the same labels again
      }
    }
  }
}

SignalContainer Digitizer::addSignalsFromPileup()
{
  return pileupTool.addSignals(mPileupSignals, mCurrentTriggerTime);
}

void Digitizer::pileup()
{
  mPileupSignals.push_back(mSignalsMapCollection);
  clearContainers();
}

void Digitizer::clearContainers()
{
  for (auto& sm : mSignalsMapCollection) {
    sm.clear();
  }
}

void Digitizer::process(std::vector<Hit> const& hits)
{
  if (!mCalib) {
    LOG(fatal) << "TRD Calibration database not available";
  }

  // Get the a hit container for all the hits in a given detector then call convertHits for a given detector (0 - 539)
  std::array<std::vector<Hit>, MAXCHAMBER> hitsPerDetector;
  getHitContainerPerDetector(hits, hitsPerDetector);

#ifdef WITH_OPENMP
// Loop over all TRD detectors (in a parallel fashion)
#pragma omp parallel for schedule(dynamic) num_threads(mNumThreads)
#endif
  for (int det = 0; det < MAXCHAMBER; ++det) {
#ifdef WITH_OPENMP
    const int threadid = omp_get_thread_num();
#else
    const int threadid = 0;
#endif
    auto& signalsMap = mSignalsMapCollection[det];
    // Jump to the next detector if the detector is
    // switched off, not installed, etc
    if (mCalib->getChamberStatus()->isNoData(det)) {
      continue;
    }
    if (!mGeo->chamberInGeometry(det)) {
      continue;
    }

    // Go to the next detector if there are no hits
    if (hitsPerDetector[det].size() == 0) {
      continue;
    }

    if (!convertHits(det, hitsPerDetector[det], signalsMap, threadid)) {
      LOG(warn) << "TRD conversion of hits failed for detector " << det;
      continue; // go to the next chamber
    }
  }
}

void Digitizer::getHitContainerPerDetector(const std::vector<Hit>& hits, std::array<std::vector<Hit>, MAXCHAMBER>& hitsPerDetector)
{
  //
  // Fill an array of size MAXCHAMBER (540)
  // The i-element of the array contains the hit collection for the i-detector
  // To be called once, before doing the loop over all detectors and process the hits
  //
  for (const auto& hit : hits) {
    hitsPerDetector[hit.GetDetectorID()].push_back(hit);
  }
}

bool Digitizer::convertHits(const int det, const std::vector<Hit>& hits, SignalContainer& signalMapCont, int thread)
{
  //
  // Convert the detector-wise sorted hits to detector signals
  //

  double padSignal[mNpad];

  const double calExBDetValue = mCalib->getExB(det); // T * V/cm (check units)
  const PadPlane* padPlane = mGeo->getPadPlane(det);
  const int layer = mGeo->getLayer(det);
  const float rowEndROC = padPlane->getRowEndROC();
  const float row0 = padPlane->getRow0ROC();
  const int nColMax = padPlane->getNcols();

  // Loop over hits
  for (const auto& hit : hits) {
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
      if ((driftLength < DrMin) || (driftLength > DrMax)) {
        continue;
      }
    }

    int rowE = padPlane->getPadRowNumberROC(locR);
    if (rowE < 0) {
      continue;
    }

    double rowOffset = padPlane->getPadRowOffsetROC(rowE, locR);
    double offsetTilt = padPlane->getTiltOffset(rowE, rowOffset);
    int colE = padPlane->getPadColNumber(locC + offsetTilt);
    if (colE < 0) {
      continue;
    }

    double absDriftLength = std::fabs(driftLength); // Normalized drift length
    if (mSimParam.isExBOn()) {
      absDriftLength /= std::sqrt(1 / (1 + calExBDetValue * calExBDetValue));
    }

    float driftVelocity = mCalib->getVDrift(det, colE, rowE); // The drift velocity
    float t0 = mCalib->getT0(det, colE, rowE);                // The T0 velocity

    // Loop over all created electrons
    const int nElectrons = std::fabs(qTotal);
    for (int el = 0; el < nElectrons; ++el) {
      // Electron attachment
      if (mSimParam.elAttachOn()) {
        if (mFlatRandomRings[thread].getNextValue() < absDriftLength * mElAttachProp) {
          continue;
        }
      }
      // scoped diffused coordinates for each electron
      double locRd{locR}, locCd{locC}, locTd{locT};

      // Apply diffusion smearing
      if (mSimParam.diffusionOn()) {
        if (!diffusion(driftVelocity, absDriftLength, calExBDetValue, locR, locC, locT, locRd, locCd, locTd, thread)) {
          continue;
        }
      }

      // Apply E x B effects
      if (mSimParam.isExBOn()) {
        locCd = locCd + calExBDetValue * driftLength;
      }
      // The electron position after diffusion and ExB in pad coordinates.
      rowE = padPlane->getPadRowNumberROC(locRd);
      if (rowE < 0) {
        continue;
      }
      rowOffset = padPlane->getPadRowOffsetROC(rowE, locRd);
      // The pad column (rphi-direction)
      offsetTilt = padPlane->getTiltOffset(rowE, rowOffset);
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
      if (mSimParam.timeStructOn()) {
        // Get z-position with respect to anode wire
        double zz = row0 - locRd + padPlane->getAnodeWireOffset();
        zz -= ((int)(2 * zz)) * 0.5;
        if (zz > 0.25) {
          zz = 0.5 - zz;
        }
        // Use drift time map (GARFIELD)
        driftTime = mDriftEstimators[thread].timeStruct(driftVelocity, 0.5 * AmWidth - 1.0 * locTd, zz, &(mFlagVdriftOutOfRange[det])) + hit.GetTime();
      } else {
        // Use constant drift velocity
        driftTime = std::fabs(locTd) / driftVelocity + hit.GetTime(); // drift time in microseconds
      }

      // Apply the gas gain including fluctuations
      const double signal = -(mSimParam.getGasGain()) * mLogRandomRings[thread].getNextValue();

      // Apply the pad response
      if (mSimParam.prfOn()) {
        // The distance of the electron to the center of the pad in units of pad width
        double dist = (colOffset - 0.5 * padPlane->getColSize(colE)) / padPlane->getColSize(colE);
        // ********************************************************************************
        // This is a fixed parametrization, i.e. not dependent on calibration values !
        // ********************************************************************************
        if (!(mPRF.getPRF(signal, dist, layer, padSignal))) {
          continue;
        }
      } else {
        padSignal[0] = 0;
        padSignal[1] = signal;
        padSignal[2] = 0;
      }

      // The time bin (always positive), with t0 distortion
      double timeBinIdeal = driftTime * mSamplingRate + t0;
      // Protection
      if (std::fabs(timeBinIdeal) > (2 * mMaxTimeBins)) { // OS: why 2*mMaxTimeBins?
        timeBinIdeal = 2 * mMaxTimeBins;
      }
      int timeBinTruncated = ((int)timeBinIdeal);
      // The distance of the position to the middle of the timebin
      double timeOffset = ((float)timeBinTruncated + 0.5 - timeBinIdeal) / mSamplingRate;
      // Sample the time response inside the drift region + additional time bins before and after.
      // The sampling is done always in the middle of the time bin
      const int firstTimeBin = std::max(timeBinTruncated, 0);
      const int lastTimeBin = std::min(timeBinTruncated + mTimeBinTRFend, mMaxTimeBins);

      // loop over pads first then over timebins for better cache friendliness
      // and less access to signalMapCont
      for (int pad = 0; pad < mNpad; ++pad) {
        int colPos = colE + pad - 1;
        if (colPos < 0) {
          continue;
        }
        if (colPos >= nColMax) {
          break;
        }

        const int key = calculateKey(det, rowE, colPos);
        auto& currentSignalData = signalMapCont[key]; // Get the old signal or make a new one if it doesn't exist
        auto& currentSignal = currentSignalData.signals;
        auto& trackIds = currentSignalData.trackIds;
        auto& labels = currentSignalData.labels;
        currentSignalData.firstTBtime = mTime;
        addLabel(hit.GetTrackID(), labels, trackIds); // add a label record only if needed

        // add signal with crosstalk for the non-central pads only
        if (colPos != colE) {
          for (int tb = firstTimeBin; tb < lastTimeBin; ++tb) {
            const double t = (tb - timeBinTruncated) / mSamplingRate + timeOffset;
            const double timeResponse = mSimParam.trfOn() ? mSimParam.timeResponse(t) : 1;
            const double crossTalk = mSimParam.ctOn() ? mSimParam.crossTalk(t) : 0;
            currentSignal[tb] += padSignal[pad] * (timeResponse + crossTalk);
          } // end of loop time bins
        } else {
          for (int tb = firstTimeBin; tb < lastTimeBin; ++tb) {
            const double t = (tb - timeBinTruncated) / mSamplingRate + timeOffset;
            const double timeResponse = mSimParam.trfOn() ? mSimParam.timeResponse(t) : 1;
            currentSignal[tb] += padSignal[pad] * timeResponse;
          } // end of loop time bins
        }
      } // end of loop over pads
    }   // end of loop over electrons
  }     // end of loop over hits
  return true;
}

void Digitizer::addLabel(const int& trackId, std::vector<o2::MCCompLabel>& labels, std::unordered_set<int>& trackIds)
{
  if (trackIds.count(trackId) == 0) {
    trackIds.insert(trackId);
    MCLabel label(trackId, getEventID(), getSrcID());
    labels.push_back(label);
  }
}

float drawGaus(o2::math_utils::RandomRing<>& normaldistRing, float mu, float sigma)
{
  // this is using standard normally distributed random numbers and rescaling to make
  // them gaussian distributed with general mu and sigma
  return mu + sigma * normaldistRing.getNextValue();
}

bool Digitizer::convertSignalsToADC(SignalContainer& signalMapCont, DigitContainer& digits, int thread)
{
  //
  // Converts the sampled electron signals to ADC values for a given chamber
  //

  constexpr double kEl2fC = 1.602e-19 * 1.0e15;                                 // Converts number of electrons to fC
  double coupling = mSimParam.getPadCoupling() * mSimParam.getTimeCoupling();   // Coupling factor
  double convert = kEl2fC * mSimParam.getChipGain();                            // Electronics conversion factor
  double adcConvert = mSimParam.getADCoutRange() / mSimParam.getADCinRange();   // ADC conversion factor
  double baseline = mSimParam.getADCbaseline() / adcConvert;                    // The electronics baseline in mV
  double baselineEl = baseline / convert;                                       // The electronics baseline in electrons

  for (auto& signalMapIter : signalMapCont) {
    const auto key = signalMapIter.first;
    const int det = getDetectorFromKey(key);
    const int row = getRowFromKey(key);
    const int col = getColFromKey(key);
    // halfchamber masking
    int mcm = (int)(col / 18);               // current group of 18 col pads
    int halfchamberside = (mcm > 3) ? 1 : 0; // 0=Aside, 1=Bside

    // Halfchambers that are switched off, masked by mCalib
    if ((halfchamberside == 0 && mCalib->getChamberStatus()->isNoDataSideA(det)) ||
        (halfchamberside == 1 && mCalib->getChamberStatus()->isNoDataSideB(det))) {
      continue;
    }

    // Check whether pad is masked
    // Bridged pads are not considered yet!!!
    if (mCalib->getPadStatus()->isMasked(det, col, row) || mCalib->getPadStatus()->isNotConnected(det, col, row)) {
      continue;
    }

    float padgain = mCalib->getPadGainFactor(det, row, col); // The gain factor
    if (padgain <= 0) {
      LOG(fatal) << "Not a valid gain " << padgain << ", " << det << ", " << col << ", " << row;
    }

    signalMapIter.second.isDigit = true; // flag the signal as digit
    // Loop over the all timebins in the ADC array
    const auto& signalArray = signalMapIter.second.signals;
    ArrayADC adcs{};
    for (int tb = 0; tb < mMaxTimeBinsTRAP; ++tb) {
      float signalAmp = (float)signalArray[tb]; // The signal amplitude
      signalAmp *= coupling;                    // Pad and time coupling
      signalAmp *= padgain;                     // Gain factors
      // Add the noise, starting from minus ADC baseline in electrons
      signalAmp = std::max((double)drawGaus(mGausRandomRings[thread], signalAmp, mSimParam.getNoise()), -baselineEl);
      signalAmp *= convert;  // Convert to mV
      signalAmp += baseline; // Add ADC baseline in mV
      // Convert to ADC counts
      // Set the overflow-bit fADCoutRange if the signal is larger than fADCinRange
      ADC_t adc = 0;
      if (signalAmp >= mSimParam.getADCinRange()) {
        adc = ((ADC_t)mSimParam.getADCoutRange());
      } else {
        adc = std::lround(signalAmp * adcConvert);
      }
      // update the adc array value
      adcs[tb] = adc;
    } // loop over timebins
    // Convert the map to digits here, and push them to the container
    digits.emplace_back(det, row, col, adcs);
    if (mCreateSharedDigits) {
      auto digit = digits.back();
      if ((digit.getChannel() == 2) && !((digit.getROB() % 2 != 0) && (digit.getMCM() % NMCMROBINCOL == 3))) {
        // shared left, if not leftmost MCM of left ROB of chamber
        int robShared = (digit.getMCM() % NMCMROBINCOL == 3) ? digit.getROB() + 1 : digit.getROB(); // for the leftmost MCM on a ROB the shared digit is added to the neighbouring ROB
        int mcmShared = (robShared == digit.getROB()) ? digit.getMCM() + 1 : digit.getMCM() - 3;
        digits.emplace_back(det, robShared, mcmShared, NADCMCM - 1, adcs);
        signalMapIter.second.isShared = true;
      } else if ((digit.getChannel() == 18 || digit.getChannel() == 19) && !((digit.getROB() % 2 == 0) && (digit.getMCM() % NMCMROBINCOL == 0))) {
        // shared right, if not rightmost MCM of right ROB of chamber
        int robShared = (digit.getMCM() % NMCMROBINCOL == 0) ? digit.getROB() - 1 : digit.getROB(); // for the rightmost MCM on a ROB the shared digit is added to the neighbouring ROB
        int mcmShared = (robShared == digit.getROB()) ? digit.getMCM() - 1 : digit.getMCM() + 3;
        digits.emplace_back(det, robShared, mcmShared, digit.getChannel() - NCOLMCM, adcs);
        signalMapIter.second.isShared = true;
      }
    }
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
  if (mDriftEstimators[thread].getDiffCoeff(diffL, diffT, vdrift)) {
    float driftSqrt = std::sqrt(absdriftlength);
    float sigmaT = driftSqrt * diffT;
    float sigmaL = driftSqrt * diffL;
    lRow = drawGaus(mGausRandomRings[thread], lRow0, sigmaT);
    if (mSimParam.isExBOn()) {
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

std::string Digitizer::dumpFlaggedChambers() const
{
  std::string retVal = "";
  for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
    if (mFlagVdriftOutOfRange[iDet]) {
      retVal += std::to_string(iDet);
      retVal += ", ";
    }
  }
  if (!retVal.empty()) {
    retVal.erase(retVal.size() - 2);
  }
  return retVal;
}
