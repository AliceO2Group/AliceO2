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

using namespace o2::trd;

Digitizer::Digitizer()
{
  // Check if you need more initialization
  o2::base::GeometryManager::loadGeometry();
  mGeo = new TRDGeometry();
  mGeo->createClusterMatrixArray(); // Requiered for chamberInGeometry()

  mPRF = new PadResponse();

  // get the Instance of simulation and common parameters
  mSimParam = TRDSimParam::Instance();
  mCommonParam = TRDCommonParam::Instance();
  // mCalib = TRDCalibDB::Instace(); // PLEASE FIX ME when CCDB is ready
  if (!mSimParam) {
    LOG(FATAL) << "TRD Simulation Parameters not available";
  }
  if (!mCommonParam) {
    LOG(FATAL) << "TRD Common Parameters not available";
  } else {
    if (!mCommonParam->cacheMagField()) {
      LOG(FATAL) << "TRD Common Parameters does not have magnetic field available";
    }
  }
  // if (!mCalib) { // PLEASE FIX ME when CCDB is ready
  //   LOG(FATAL) << "TRD mCalib database not available";
  // }
  mSDigits = false;
}

void Digitizer::process(std::vector<HitType> const& hits, DigitContainer_t& digitCont, o2::dataformats::MCTruthContainer<MCLabel>& labels)
{
  // (WIP) Implementation for digitization

  // Check if Geometry and if CCDB are available as they will be requiered
  // const int nTimeBins = mCalib->GetNumberOfTimeBinsDCS(); PLEASE FIX ME when CCDB is ready

  SignalContainer_t adcMapCont;
  mLabels.clear();

  // Get the a hit container for all the hits in a given detector then call convertHits for a given detector (0 - 539)
  std::array<std::vector<HitType>, kNdet> hitsPerDetector;
  getHitContainerPerDetector(hits, hitsPerDetector);

  // Loop over all TRD detectors
  for (int det = 0; det < kNdet; ++det) {
    // Jump to the next detector if the detector is
    // switched off, not installed, etc
    /*      
    if (mCalib->IsChamberNoData(det)) { // PLEASE FIX ME when CCDB is ready
      continue;
    } */
    if (!mGeo->chamberInGeometry(det)) {
      continue;
    }

    // Go to the next detector if there are no hits
    if (hitsPerDetector[det].size() == 0) {
      continue;
    }

    if (!convertHits(det, hitsPerDetector[det], adcMapCont)) {
      LOG(WARN) << "TRD conversion of hits failed for detector " << det;
      continue; // go to the next chamber
    }

    if (!convertSignalsToDigits(det, adcMapCont)) {
      LOG(WARN) << "TRD conversion of signals to digits failed for detector " << det;
      continue; // go to the next chamber
    }
  }

  // Finalize
  Digit::convertMapToVectors(adcMapCont, digitCont);

  // MC labels
  for (int i = 0; i < mLabels.size(); ++i) {
    labels.addElement(i, mLabels[i]);
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

bool Digitizer::convertHits(const int det, const std::vector<HitType>& hits, SignalContainer_t& adcMapCont)
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
  double signalOld[kNpad];

  // Get the detector wise mCalib objects
  // const TRDCalDet* calVdriftDet = mCalib->GetVdriftDet();    PLEASE FIX ME when CCDB is ready
  // const TRDCalDet* calT0Det = mCalib->GetT0Det();            PLEASE FIX ME when CCDB is ready
  // const TRDCalDet* calExBDet = mCalib->GetExBDet();          PLEASE FIX ME when CCDB is ready

  // FIX ME: Default values until I have implemented the mCalib objects
  //
  // See Table 8 (Nuclear Inst. and Methods in Physics Research, A 881 (2018) 88-127)
  // Defaults values  from OCDB (AliRoot DrawTrending macro - Thanks to Y. Pachmayer)
  // For 5 TeV pp - 27 runs from LHC15n
  //
  float calVdriftDetValue = 1.48; // cm/microsecond         // calVdriftDet->GetValue(det); PLEASE FIX ME when CCDB is ready
  float calT0DetValue = -1.38;    // microseconds           // calT0Det->GetValue(det);     PLEASE FIX ME when CCDB is ready
  double calExBDetValue = 0.16;   // T * V/cm (check units) // calExBDet->GetValue(det);    PLEASE FIX ME when CCDB is ready

  // TRDCalROC* calVdriftROC = mCalib->GetVdriftROC(det); PLEASE FIX ME when CCDB is ready
  // TRDCalROC* calT0ROC = mCalib->GetT0ROC(det);         PLEASE FIX ME when CCDB is ready

  if (mSimParam->TRFOn()) {
    timeBinTRFend = ((int)(mSimParam->GetTRFhi() * mCommonParam->GetSamplingFrequency())) - 1;
  }

  const int nTimeTotal = kTimeBins; // PLEASE FIX ME when CCDB is ready
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
    double locC = hit.getLocalC(); // col direction in amplification or drift volume
    double locR = hit.getLocalR(); // row direction in amplification or drift volume
    double locT = hit.getLocalT(); // time direction in amplification or drift volume

    if (hit.isFromDriftRegion()) {
      locT = locT - kDrWidth / 2 - kAmWidth / 2;
    }

    const double driftLength = -1 * locT; // The drift length in cm without diffusion

    // Patch to take care of TR photons that are absorbed
    // outside the chamber volume. A real fix would actually need
    // a more clever implementation of the TR hit generation
    if (qTotal < 0) {
      if ((locR < rowEndROC) ||
          (locR > row0)) {
        continue;
      }
      if ((driftLength < kDrMin) ||
          (driftLength > kDrMax)) {
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

    // FIX ME: Commented out what is still not yet implemented
    double absDriftLength = abs(driftLength); // Normalized drift length
    if (mCommonParam->ExBOn()) {
      absDriftLength /= TMath::Sqrt(1 / (1 + calExBDetValue * calExBDetValue));
    }
    // double driftVelocity = calVdriftDetValue * calVdriftROC->GetValue(colE, rowE); PLEASE FIX ME when CCDB is ready
    double driftVelocity = 2.13; // Defaults values  from OCDB (AliRoot DrawTrending macro) for 5 TeV pp - 27 runs from LHC15n

    // Loop over all created electrons
    const int nElectrons = abs(qTotal);
    for (int el = 0; el < nElectrons; ++el) {
      // Electron attachment
      if (mSimParam->ElAttachOn()) {
        if (gRandom->Rndm() < absDriftLength * elAttachProp) {
          continue;
        }
      }
      // scoped diffused coordinates for each electron
      double locRd{locR}, locCd{locC}, locTd{locT};

      // Apply diffusion smearing
      if (mSimParam->DiffusionOn()) {
        if (!diffusion(driftVelocity, absDriftLength, calExBDetValue, locR, locC, locT, locRd, locCd, locTd)) {
          continue;
        }
      }
      // Apply E x B effects
      if (mCommonParam->ExBOn()) {
        locCd = locCd + calExBDetValue * driftLength;
      }
      // The electron position after diffusion and ExB in pad coordinates.
      rowE = padPlane->getPadRowNumberROC(locRd);
      if (rowE < 1) {
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
      // Retrieve drift velocity becuase col and row may have changed
      // driftVelocity = calVdriftDetValue* calVdriftROC->GetValue(colE, rowE);  PLEASE FIX ME when CCDB is ready
      driftVelocity = 2.13; // Defaults values  from OCDB (AliRoot DrawTrending macro) for 5 TeV pp - 27 runs from LHC15n
      // float t0 = calT0DetValue + calT0ROC->getValue(colE, rowE);      PLEASE FIX ME when CCDB is ready
      const float t0 = -1.38 + 0; // Defaults values  from OCDB (AliRoot DrawTrending macro) for 5 TeV pp - 27 runs from LHC15n
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
        driftTime = mCommonParam->TimeStruct(driftVelocity, 0.5 * kAmWidth - 1.0 * locTd, zz) + hit.GetTime();
      } else {
        // Use constant drift velocity
        driftTime = abs(locTd) / driftVelocity + hit.GetTime();
      }

      // Apply the gas gain including fluctuations
      double ggRndm = 0;
      do {
        ggRndm = gRandom->Rndm();
      } while (ggRndm <= 0);
      double signal = -(mSimParam->GetGasGain()) * TMath::Log(ggRndm);

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
      if (abs(timeBinIdeal) > 2 * nTimeTotal) {
        timeBinIdeal = 2 * nTimeTotal;
      }
      int timeBinTruncated = ((int)timeBinIdeal);
      // The distance of the position to the middle of the timebin
      double timeOffset = ((float)timeBinTruncated + 0.5 - timeBinIdeal) / samplingRate;

      // Sample the time response inside the drift region + additional time bins before and after.
      // The sampling is done always in the middle of the time bin
      const int firstTimeBin = TMath::Max(timeBinTruncated, 0);
      const int lastTimeBin = TMath::Min(timeBinTruncated + timeBinTRFend, nTimeTotal);
      for (int iTimeBin = firstTimeBin; iTimeBin < lastTimeBin; ++iTimeBin) {
        // Apply the time response
        double timeResponse = 1;
        double crossTalk = 0;
        const double t = (iTimeBin - timeBinTruncated) / samplingRate + timeOffset;
        if (mSimParam->TRFOn()) {
          timeResponse = mSimParam->TimeResponse(t);
        }
        if (mSimParam->CTOn()) {
          crossTalk = mSimParam->CrossTalk(t);
        }
        signalOld[0] = 0;
        signalOld[1] = 0;
        signalOld[2] = 0;
        for (int iPad = 0; iPad < kNpad; iPad++) {
          int colPos = colE + iPad - 1;
          if (colPos < 0) {
            continue;
          }
          if (colPos >= nColMax) {
            break;
          }
          // Add the signals
          // Get the old signal
          const int key = Digit::calculateKey(det, rowE, colPos);
          if (key < KEY_MIN || key > KEY_MAX) {
            LOG(FATAL) << "Wrong TRD key " << key << " for (det,row,col) = (" << det << ", " << rowE << ", " << colPos << ")";
          }

          signalOld[iPad] = adcMapCont[key][iTimeBin];
          if (colPos != colE) {
            // Cross talk added to non-central pads
            signalOld[iPad] += padSignal[iPad] * (timeResponse + crossTalk);
          } else {
            // Without cross talk at central pad
            signalOld[iPad] += padSignal[iPad] * timeResponse;
          }
          // Update the final signal
          adcMapCont[key][iTimeBin] = signalOld[iPad];
          isDigit = true;
        } // Loop: pads
      }   // Loop: time bins
    }     // end of loop over electrons
    mLabels.emplace_back(hit.GetTrackID(), mEventID, mSrcID, isDigit);
  } // end of loop over hits
  return true;
}

bool Digitizer::convertSignalsToDigits(const int det, SignalContainer_t& adcMapCont)
{
  //
  // conversion of signals to digits
  //

  if (mSDigits) {
    // Convert the signal array to s-digits
    if (!convertSignalsToSDigits(det, adcMapCont)) {
      return false;
    }
  } else {
    // Convert the signal array to digits
    if (!convertSignalsToADC(det, adcMapCont)) {
      return false;
    }
    // Run digital processing for digits
    // RunDigitalProcessing(det);
  }
  return true;
}

bool Digitizer::convertSignalsToSDigits(const int det, SignalContainer_t& adcMapCont)
{
  //
  // Convert signals to S-digits
  //
  LOG(FATAL) << "You shouldn't be here. This is not implemented yet.";
  return false;
}

bool Digitizer::convertSignalsToADC(const int det, SignalContainer_t& adcMapCont)
{
  //
  // Converts the sampled electron signals to ADC values for a given chamber
  //
  if (adcMapCont.size() == 0) {
    return false;
  }

  constexpr double kEl2fC = 1.602e-19 * 1.0e15;                                 // Converts number of electrons to fC
  double coupling = mSimParam->GetPadCoupling() * mSimParam->GetTimeCoupling(); // Coupling factor
  double convert = kEl2fC * mSimParam->GetChipGain();                           // Electronics conversion factor
  double adcConvert = mSimParam->GetADCoutRange() / mSimParam->GetADCinRange(); // ADC conversion factor
  double baseline = mSimParam->GetADCbaseline() / adcConvert;                   // The electronics baseline in mV
  double baselineEl = baseline / convert;                                       // The electronics baseline in electrons

  int nRowMax = mGeo->getPadPlane(det)->getNrows();
  int nColMax = mGeo->getPadPlane(det)->getNcols();
  int nTimeTotal = kTimeBins; // fDigitsManager->GetDigitsParam()->GetNTimeBins(det);

  // Get the mCalib objects
  // CalDet* calGainFactorDet = mCalib->GetGainFactorDet();
  // CalRoc* calGainFactorROC = mCalib->GetGainFactorROC(det);
  // calGainFactorDetValue = calGainFactorDet->GetValue(det);
  float calGainFactorDetValue = 0.47; // +/- 0.06 // Defaults value  from OCDB (AliRoot DrawTrending macro) for 5 TeV pp - 27 runs from LHC15n

  // Create the digits for this chamber
  // for (int row = 0; row < nRowMax; row++) {
  //   for (int col = 0; col < nColMax; col++) {
  for (auto& adcMapIter : adcMapCont) {
    const int row = Digit::getRowFromKey(adcMapIter.first); // for the next line, when ccdb is ready
    const int col = Digit::getColFromKey(adcMapIter.first); // for the next line, when ccdb is ready
    // halfchamber masking
    int iMcm = (int)(col / 18);               // current group of 18 col pads
    int halfchamberside = (iMcm > 3 ? 1 : 0); // 0=Aside, 1=Bside
    // Halfchambers that are switched off, masked by mCalib
    // if (mCalib->IsHalfChamberNoData(det, halfchamberside))
    //   continue;
    // Check whether pad is masked
    // Bridged pads are not considered yet!!!
    // if (mCalib->IsPadMasked(det, col, row) ||
    //     mCalib->IsPadNotConnected(det, col, row)) {
    //   continue;
    // }

    // The gain factors
    float padgain = calGainFactorDetValue; // * calGainFactorROC->GetValue(col, row); // PLEASE FIX ME when CCDB is ready
    if (padgain <= 0) {
      LOG(FATAL) << "Not a valid gain " << padgain
                 << ", " << det
                 << ", " << col
                 << ", " << row;
    }
    // loop over time bins
    // for (int tb = 0; tb < nTimeTotal; tb++) {
    for (auto& adcArrayVal : adcMapIter.second) {
      float signalAmp = (float)adcArrayVal; // The signal amplitude
      signalAmp *= coupling;                // Pad and time coupling
      signalAmp *= padgain;                 // Gain factors
      // Add the noise, starting from minus ADC baseline in electrons
      signalAmp = TMath::Max((double)gRandom->Gaus(signalAmp, mSimParam->GetNoise()), -baselineEl);
      signalAmp *= convert;  // Convert to mV
      signalAmp += baseline; // Add ADC baseline in mV
      // Convert to ADC counts
      // Set the overflow-bit fADCoutRange if the signal is larger than fADCinRange
      ADC_t adc = 0;
      if (signalAmp >= mSimParam->GetADCinRange()) {
        adc = ((ADC_t)mSimParam->GetADCoutRange());
      } else {
        adc = TMath::Nint(signalAmp * adcConvert);
      }
      // update the adc array value
      adcArrayVal = adc;
    } // for: tb
  }
  return true;
}

bool Digitizer::diffusion(float vdrift, double absdriftlength, double exbvalue,
                          double lRow0, double lCol0, double lTime0,
                          double& lRow, double& lCol, double& lTime)
{
  //
  // Applies the diffusion smearing to the position of a single electron.
  // Depends on absolute drift length.
  //
  float diffL = 0.0;
  float diffT = 0.0;
  if (mCommonParam->GetDiffCoeff(diffL, diffT, vdrift)) {
    float driftSqrt = TMath::Sqrt(absdriftlength);
    float sigmaT = driftSqrt * diffT;
    float sigmaL = driftSqrt * diffL;
    lRow = gRandom->Gaus(lRow0, sigmaT);
    if (mCommonParam->ExBOn()) {
      lCol = gRandom->Gaus(lCol0, sigmaT * 1.0 / (1.0 + exbvalue * exbvalue));
      lTime = gRandom->Gaus(lTime0, sigmaL * 1.0 / (1.0 + exbvalue * exbvalue));
    } else {
      lCol = gRandom->Gaus(lCol0, sigmaT);
      lTime = gRandom->Gaus(lTime0, sigmaL);
    }
    return true;
  } else {
    return false;
  }
}
