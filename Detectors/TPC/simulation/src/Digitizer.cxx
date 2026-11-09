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

/// \file Digitizer.cxx
/// \brief Implementation of the ALICE TPC digitizer
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TH3.h"

#include "TPCSimulation/Digitizer.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterGEM.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCBaseRecSim/CDBInterface.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/CorrMapParam.h"

#include <fairlogger/Logger.h>
#include <TRandom3.h>

ClassImp(o2::tpc::Digitizer);

using namespace o2::tpc;

Digitizer::~Digitizer() = default;

Digitizer::Digitizer() = default;

namespace
{
// splitmix64 finaliser, to derive well separated seeds from the base seed
ULong64_t mixSeed(ULong64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// TRandom3 takes 32 bits of the seed and treats 0 as a request for a random seed
UInt_t toTRandom3Seed(ULong64_t x)
{
  const UInt_t seed = static_cast<UInt_t>(x ^ (x >> 32));
  return seed ? seed : 1;
}
} // namespace

ElectronTransport& Digitizer::electronTransport()
{
  return mElectronTransport ? *mElectronTransport : ElectronTransport::instance();
}

GEMAmplification& Digitizer::gemAmplification()
{
  return mGEMAmplification ? *mGEMAmplification : GEMAmplification::instance();
}

SAMPAProcessing& Digitizer::sampaProcessing()
{
  return mSAMPAProcessing ? *mSAMPAProcessing : SAMPAProcessing::instance();
}

void Digitizer::setUseSectorSeeding(ULong64_t seed)
{
  mUseSectorSeeding = true;
  mSeed = seed;
}

void Digitizer::init()
{
  if (mUseSectorSeeding) {
    if (!mGEMAmplification) {
      // the rings are filled once, in the construction order of the singletons
      TRandom3 rng(toTRandom3Seed(mixSeed(mSeed)));
      mGEMAmplification = std::make_unique<GEMAmplification>(&rng);
      mElectronTransport = std::make_unique<ElectronTransport>(&rng);
      mSAMPAProcessing = std::make_unique<SAMPAProcessing>(&rng);
      mDigitContainer.setSAMPAProcessing(mSAMPAProcessing.get());
    }
    const ULong64_t sectorSeed = mixSeed(mSeed ^ mixSeed(static_cast<ULong64_t>(mSector) + 1));
    TRandom3 rng(toTRandom3Seed(mixSeed(sectorSeed ^ mixSeed(mSeedingTimeframe))));
    mGEMAmplification->setRandomRingPositions(rng);
    mElectronTransport->setRandomRingPositions(rng);
    mSAMPAProcessing->setRandomRingPositions(rng);
  }
  auto& gemAmplification = this->gemAmplification();
  gemAmplification.updateParameters();
  auto& electronTransport = this->electronTransport();
  electronTransport.updateParameters(mVDrift);
  auto& sampaProcessing = this->sampaProcessing();
  // the calibration update may reload FEE parameters from file, so SAMPAProcessing picks up its pointers afterwards
  mDigitContainer.updateCalibrations();
  sampaProcessing.updateParameters(mVDrift);
}

void Digitizer::process(const std::vector<o2::tpc::HitGroup>& hits,
                        const int eventID, const int sourceID)
{
  const Mapper& mapper = Mapper::instance();
  auto& detParam = ParameterDetector::Instance();
  auto& eleParam = ParameterElectronics::Instance();
  auto& gemParam = ParameterGEM::Instance();

  auto& gemAmplification = this->gemAmplification();
  auto& electronTransport = this->electronTransport();
  auto& sampaProcessing = this->sampaProcessing();

  const int nShapedPoints = eleParam.NShapedPoints;
  const auto amplificationMode = gemParam.AmplMode;
  mSignalArray.resize(nShapedPoints);

  /// Reserve space in the digit container for the current event
  mDigitContainer.reserve(sampaProcessing.getTimeBinFromTime(mEventTime - mOutputDigitTimeOffset));

  /// obtain max drift_time + hitTime which can be processed
  float maxEleTime = (int(mDigitContainer.size()) - nShapedPoints) * eleParam.ZbinWidth;

  for (auto& hitGroup : hits) {
    const int MCTrackID = hitGroup.GetTrackID();
    for (size_t hitindex = 0; hitindex < hitGroup.getSize(); ++hitindex) {
      const auto& eh = hitGroup.getHit(hitindex);

      GlobalPosition3D posEle(eh.GetX(), eh.GetY(), eh.GetZ());

      // Distort the electron position in case space-charge distortions are used
      if (mDistortionScaleType == 1) {
        mSpaceCharge->distortElectron(posEle);
      } else if (mDistortionScaleType == 2) {
        mSpaceCharge->distortElectron(posEle, (mUseScaledDistortions ? nullptr : mSpaceChargeDer), mLumiScaleFactor);
      }

      /// Remove electrons that end up more than three sigma of the hit's average diffusion away from the current sector
      /// boundary
      if (electronTransport.isCompletelyOutOfSectorCoarseElectronDrift(posEle, mSector)) {
        continue;
      }

      /// The energy loss stored corresponds to nElectrons
      const int nPrimaryElectrons = static_cast<int>(eh.GetEnergyLoss());
      const float hitTime = eh.GetTime() * 0.001; /// in us
      float driftTime = 0.f;

      /// TODO: add primary ions to space-charge density

      /// Loop over electrons
      for (int iEle = 0; iEle < nPrimaryElectrons; ++iEle) {

        /// Drift and Diffusion
        const GlobalPosition3D posEleDiff = electronTransport.getElectronDrift(posEle, driftTime);
        const float eleTime = driftTime + hitTime; /// in us
        if (eleTime >= maxEleTime) {
          // LOG(warning) << "Skipping electron with driftTime " << driftTime << " from hit at time " << hitTime;
          continue;
        }
        const float absoluteTime = eleTime + mTDriftOffset + (mEventTime - mOutputDigitTimeOffset); /// in us

        /// the absolute time needs to be within the readout limits
        /// (otherwise negative times would all be accumulated in the 0-th timebin further below)
        if (!(absoluteTime >= 0 /* && absoluteTime <= timeframelength */)) {
          continue;
        }

        /// Attachment
        if (electronTransport.isElectronAttachment(driftTime)) {
          continue;
        }

        /// Remove electrons that end up outside the active volume
        if (std::abs(posEleDiff.Z()) > detParam.TPClength) {
          continue;
        }

        /// When the electron is not in the sector we're processing, abandon
        if (mapper.isOutOfSector(posEleDiff, mSector)) {
          continue;
        }

        /// Compute digit position and check for validity
        const DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(posEleDiff, mSector);
        if (!digiPadPos.isValid()) {
          continue;
        }

        /// Remove digits the end up outside the currently produced sector
        if (digiPadPos.getCRU().sector() != mSector) {
          continue;
        }

        /// Electron amplification
        const int nElectronsGEM = gemAmplification.getStackAmplification(digiPadPos.getCRU(), digiPadPos.getPadPos(), amplificationMode);
        if (nElectronsGEM == 0) {
          continue;
        }

        const GlobalPadNumber globalPad = mapper.globalPadNumber(digiPadPos.getGlobalPadPos());
        const float ADCsignal = sampaProcessing.getADCvalue(static_cast<float>(nElectronsGEM));
        const MCCompLabel label(MCTrackID, eventID, sourceID, false);
        sampaProcessing.getShapedSignal(ADCsignal, absoluteTime, mSignalArray);
        for (float i = 0; i < nShapedPoints; ++i) {
          const float time = absoluteTime + i * eleParam.ZbinWidth;
          mDigitContainer.addDigit(label, digiPadPos.getCRU(), sampaProcessing.getTimeBinFromTime(time), globalPad,
                                   mSignalArray[i]);
        }
        /// TODO: add ion backflow to space-charge density
      }
      /// end of loop over electrons
    }
  }
}

void Digitizer::flush(std::vector<o2::tpc::Digit>& digits,
                      o2::dataformats::MCTruthContainer<o2::MCCompLabel>& labels,
                      std::vector<o2::tpc::CommonMode>& commonModeOutput,
                      bool finalFlush)
{
  SAMPAProcessing& sampaProcessing = this->sampaProcessing();
  mDigitContainer.fillOutputContainer(digits, labels, commonModeOutput, mSector, sampaProcessing.getTimeBinFromTime(mEventTime - mOutputDigitTimeOffset), mIsContinuous, finalFlush);
  // flushing debug output to file
  if (((finalFlush && mIsContinuous) || (!mIsContinuous)) && mSpaceCharge) {
    o2::utils::DebugStreamer::instance()->flush();
  }
}

void Digitizer::setUseSCDistortions(const SCDistortionType& distortionType, const TH3* hisInitialSCDensity)
{
  mUseSCDistortions = true;
  if (!mSpaceCharge) {
    mSpaceChargeOwned = std::make_unique<SC>();
    mSpaceCharge = mSpaceChargeOwned.get();
  }
  mSpaceCharge->setSCDistortionType(distortionType);
  if (hisInitialSCDensity) {
    mSpaceCharge->fillChargeDensityFromHisto(*hisInitialSCDensity);
    mSpaceCharge->setUseInitialSCDensity(true);
  }
}

void Digitizer::setUseSCDistortions(SC* spaceCharge)
{
  mUseSCDistortions = true;
  if (spaceCharge != mSpaceChargeOwned.get()) {
    mSpaceChargeOwned.reset();
  }
  mSpaceCharge = spaceCharge;
  mSpaceCharge->initAfterReadingFromFile();
  mSpaceCharge->printMetaData();
}

void Digitizer::setSCDistortionsDerivative(SC* spaceCharge)
{
  mSpaceChargeDer = spaceCharge;
  mSpaceChargeDer->initAfterReadingFromFile();
  mSpaceChargeDer->printMetaData();
}

void Digitizer::setUseSCDistortions(std::string_view finp)
{
  mUseSCDistortions = true;
  if (!mSpaceCharge) {
    mSpaceChargeOwned = std::make_unique<SC>();
    mSpaceCharge = mSpaceChargeOwned.get();
  }

  // in case analytical distortions are loaded from file they are applied
  mSpaceCharge->setAnalyticalCorrectionsDistortionsFromFile(finp);
  if (!mSpaceCharge->getUseAnalyticalDistCorr()) {
    mSpaceCharge->setGlobalDistortionsFromFile(finp, Side::A);
    mSpaceCharge->setGlobalDistortionsFromFile(finp, Side::C);
  }
}

void Digitizer::setStartTime(double time)
{
  // this is setting the first timebin index for the digit container
  // note that negative times w.r.t start of timeframe/data-taking == mOutputDigitTimeOffset
  // will yield the 0-th bin (due to casting logic in sampaProcessing)
  SAMPAProcessing& sampaProcessing = this->sampaProcessing();
  sampaProcessing.updateParameters(mVDrift);
  const auto timediff = time - mOutputDigitTimeOffset;
  const auto starttimebin = sampaProcessing.getTimeBinFromTime(timediff);
  mDigitContainer.setStartTime(starttimebin);
}

void Digitizer::setLumiScaleFactor()
{
  mLumiScaleFactor = (CorrMapParam::Instance().lumiInst - mSpaceCharge->getMeanLumi()) / mSpaceChargeDer->getMeanLumi();
  LOGP(info, "Setting Lumi scale factor: lumiInst: {}  lumi mean: {} lumi mean derivative: {} lumi scale factor: {}", CorrMapParam::Instance().lumiInst, mSpaceCharge->getMeanLumi(), mSpaceChargeDer->getMeanLumi(), mLumiScaleFactor);
}

void Digitizer::setMeanLumiDistortions(float meanLumi)
{
  mSpaceCharge->setMeanLumi(meanLumi);
}

void Digitizer::setMeanLumiDistortionsDerivative(float meanLumi)
{
  mSpaceChargeDer->setMeanLumi(meanLumi);
}

void Digitizer::recalculateDistortions()
{
  if (!mSpaceCharge || !mSpaceChargeDer) {
    LOGP(info, "Average or derivative distortions not set");
    return;
  }

  // recalculate distortions only in case the inst lumi differs from the avg lumi
  if (mSpaceCharge->getMeanLumi() != CorrMapParam::Instance().lumiInst) {
    for (int iside = 0; iside < 2; ++iside) {
      const o2::tpc::Side side = (iside == 0) ? Side::A : Side::C;
      // this needs to be done only once
      LOGP(info, "Calculating corrections for average distortions");
      mSpaceCharge->calcGlobalCorrWithGlobalDistIterative(side, nullptr, 0);

      LOGP(info, "Calculating corrections for derivative distortions");
      mSpaceChargeDer->calcGlobalCorrWithGlobalDistIterative(side, nullptr, 0);

      LOGP(info, "Calculating scaled distortions with scaling factor {}", mLumiScaleFactor);
      mSpaceCharge->calcGlobalDistWithGlobalCorrIterativeLinearCartesian(side, mSpaceChargeDer, mLumiScaleFactor);
    }
    // set new lumi of avg map
    mSpaceCharge->setMeanLumi(CorrMapParam::Instance().lumiInst);
  } else {
    LOGP(info, "Inst. lumi {} is same as mean lumi {}. Skip recalculation of distortions", CorrMapParam::Instance().lumiInst, mSpaceCharge->getMeanLumi());
  }

  mUseScaledDistortions = true;
}
