// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCBase/CDBInterface.h"

#include "TPCBase/Mapper.h"

#include "FairLogger.h"

ClassImp(o2::tpc::Digitizer);

using namespace o2::tpc;

bool o2::tpc::Digitizer::mIsContinuous = true;

void Digitizer::init()
{
  // Calculate distortion lookup tables if initial space-charge density is provided
  if (mUseSCDistortions) {
    mSpaceCharge->init();
  }
}

void Digitizer::process(const std::vector<o2::tpc::HitGroup>& hits,
                        const int eventID, const int sourceID)
{
  const static Mapper& mapper = Mapper::instance();
  auto& detParam = ParameterDetector::Instance();
  auto& eleParam = ParameterElectronics::Instance();
  auto& gemParam = ParameterGEM::Instance();

  static GEMAmplification& gemAmplification = GEMAmplification::instance();
  gemAmplification.updateParameters();
  static ElectronTransport& electronTransport = ElectronTransport::instance();
  electronTransport.updateParameters();
  static SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  sampaProcessing.updateParameters();

  const int nShapedPoints = eleParam.NShapedPoints;
  const auto amplificationMode = gemParam.AmplMode;
  static std::vector<float> signalArray;
  signalArray.resize(nShapedPoints);

  /// Reserve space in the digit container for the current event
  mDigitContainer.reserve(sampaProcessing.getTimeBinFromTime(mEventTime));

  /// obtain max drift_time + hitTime which can be processed
  float maxEleTime = (int(mDigitContainer.size()) - nShapedPoints) * eleParam.ZbinWidth;

  for (auto& hitGroup : hits) {
    const int MCTrackID = hitGroup.GetTrackID();
    for (size_t hitindex = 0; hitindex < hitGroup.getSize(); ++hitindex) {
      const auto& eh = hitGroup.getHit(hitindex);

      GlobalPosition3D posEle(eh.GetX(), eh.GetY(), eh.GetZ());

      // Distort the electron position in case space-charge distortions are used
      if (mUseSCDistortions) {
        mSpaceCharge->distortElectron(posEle);
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
        if (eleTime > maxEleTime) {
          LOG(WARNING) << "Skipping electron with driftTime " << driftTime << " from hit at time " << hitTime;
          continue;
        }
        const float absoluteTime = eleTime + mEventTime; /// in us

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
  static SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  mDigitContainer.fillOutputContainer(digits, labels, commonModeOutput, mSector, sampaProcessing.getTimeBinFromTime(mEventTime), mIsContinuous, finalFlush);
}

void Digitizer::setUseSCDistortions(SpaceCharge::SCDistortionType distortionType, TH3* hisInitialSCDensity, int nZSlices, int nPhiBins, int nRBins)
{
  mUseSCDistortions = true;
  if (!mSpaceCharge) {
    mSpaceCharge = std::make_unique<SpaceCharge>(nRBins, nPhiBins, nZSlices);
  }
  mSpaceCharge->setSCDistortionType(distortionType);
  if (hisInitialSCDensity) {
    mSpaceCharge->setInitialSpaceChargeDensity(hisInitialSCDensity);
  }
}

void Digitizer::setUseSCDistortions(SpaceCharge* spaceCharge)
{
  mUseSCDistortions = true;
  mSpaceCharge.reset(spaceCharge);
}
