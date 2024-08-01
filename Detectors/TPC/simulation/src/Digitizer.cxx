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
#include "TPCBase/CDBInterface.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include "TPCBase/Mapper.h"
#include "TPCCalibration/CorrMapParam.h"

#include <fairlogger/Logger.h>

ClassImp(o2::tpc::Digitizer);

using namespace o2::tpc;

Digitizer::~Digitizer() = default;

Digitizer::Digitizer() = default;

void Digitizer::init()
{
  auto& gemAmplification = GEMAmplification::instance();
  gemAmplification.updateParameters();
  auto& electronTransport = ElectronTransport::instance();
  electronTransport.updateParameters(mVDrift);
  auto& sampaProcessing = SAMPAProcessing::instance();
  sampaProcessing.updateParameters(mVDrift);
}

void Digitizer::process(const std::vector<o2::tpc::HitGroup>& hits,
                        const int eventID, const int sourceID)
{
  const Mapper& mapper = Mapper::instance();
  auto& detParam = ParameterDetector::Instance();
  auto& eleParam = ParameterElectronics::Instance();
  auto& gemParam = ParameterGEM::Instance();

  auto& gemAmplification = GEMAmplification::instance();
  auto& electronTransport = ElectronTransport::instance();
  auto& sampaProcessing = SAMPAProcessing::instance();

  const int nShapedPoints = eleParam.NShapedPoints;
  const auto amplificationMode = gemParam.AmplMode;
  static std::vector<float> signalArray;
  signalArray.resize(nShapedPoints);

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
        mSpaceCharge->distortElectron(posEle, (mUseScaledDistortions ? nullptr : mSpaceChargeDer.get()), mLumiScaleFactor);
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
        sampaProcessing.getShapedSignal(ADCsignal, absoluteTime, signalArray);
        for (float i = 0; i < nShapedPoints; ++i) {
          const float time = absoluteTime + i * eleParam.ZbinWidth;
          mDigitContainer.addDigit(label, digiPadPos.getCRU(), sampaProcessing.getTimeBinFromTime(time), globalPad,
                                   signalArray[i]);
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
  SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
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
    mSpaceCharge = std::make_unique<SC>();
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
  mSpaceCharge.reset(spaceCharge);
  mSpaceCharge->initAfterReadingFromFile();
  mSpaceCharge->printMetaData();
}

void Digitizer::setSCDistortionsDerivative(SC* spaceCharge)
{
  mSpaceChargeDer.reset(spaceCharge);
  mSpaceChargeDer->initAfterReadingFromFile();
  mSpaceChargeDer->printMetaData();
}

void Digitizer::setUseSCDistortions(std::string_view finp)
{
  mUseSCDistortions = true;
  if (!mSpaceCharge) {
    mSpaceCharge = std::make_unique<SC>();
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
  SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  sampaProcessing.updateParameters(mVDrift);
  mDigitContainer.setStartTime(sampaProcessing.getTimeBinFromTime(time - mOutputDigitTimeOffset));
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
  if (!mSpaceCharge || !mSpaceCharge) {
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
      mSpaceCharge->calcGlobalDistWithGlobalCorrIterative(side, mSpaceChargeDer.get(), mLumiScaleFactor);
    }
    // set new lumi of avg map
    mSpaceCharge->setMeanLumi(CorrMapParam::Instance().lumiInst);
  } else {
    LOGP(info, "Inst. lumi {} is same as mean lumi {}. Skip recalculation of distortions", CorrMapParam::Instance().lumiInst, mSpaceCharge->getMeanLumi());
  }

  mUseScaledDistortions = true;
}
