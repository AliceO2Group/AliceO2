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

#include "TPCSimulation/Digitizer.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/SAMPAProcessing.h"

#include "TPCBase/Mapper.h"

#include "FairLogger.h"

ClassImp(o2::TPC::Digitizer)

  using namespace o2::TPC;

bool o2::TPC::Digitizer::mIsContinuous = true;

Digitizer::Digitizer() : mDigitContainer(nullptr) {}

Digitizer::~Digitizer() { delete mDigitContainer; }

void Digitizer::init() { mDigitContainer = new DigitContainer(); }

DigitContainer* Digitizer::Process(const Sector& sector, const std::vector<o2::TPC::HitGroup>& hits, int eventID,
                                   float eventTime)
{
  const static Mapper& mapper = Mapper::instance();
  const static ParameterDetector& detParam = ParameterDetector::defaultInstance();
  const static ParameterElectronics& eleParam = ParameterElectronics::defaultInstance();

  if (!mIsContinuous) {
    eventTime = 0.f; /// transform in us
  }

  /// \todo static_thread for thread savety?
  static GEMAmplification gemAmplification;
  static ElectronTransport electronTransport;
  static PadResponse padResponse;

  const int nShapedPoints = eleParam.getNShapedPoints();
  static std::vector<float> signalArray;
  signalArray.resize(nShapedPoints);

  for (auto& inputgroup : hits) {
    //    auto *inputgroup = static_cast<HitGroup*>(pointObject);
    const int MCTrackID = inputgroup.GetTrackID();
    for (size_t hitindex = 0; hitindex < inputgroup.getSize(); ++hitindex) {
      const auto& eh = inputgroup.getHit(hitindex);

      const GlobalPosition3D posEle(eh.GetX(), eh.GetY(), eh.GetZ());

      // The energy loss stored is really nElectrons
      const int nPrimaryElectrons = static_cast<int>(eh.GetEnergyLoss());

      /// Loop over electrons
      /// \todo can be vectorized?
      /// \todo split transport and signal formation in two separate loops?
      for (int iEle = 0; iEle < nPrimaryElectrons; ++iEle) {

        /// Drift and Diffusion
        float driftTime = 0.f;
        const GlobalPosition3D posEleDiff = electronTransport.getElectronDrift(posEle, driftTime);
        const float absoluteTime = driftTime + eventTime + eh.GetTime() * 0.001; /// in us

        /// Attachment
        if (electronTransport.isElectronAttachment(driftTime)) {
          continue;
        }

        /// Remove electrons that end up outside the active volume
        /// \todo should go to mapper?
        if (std::abs(posEleDiff.Z()) > detParam.getTPClength()) {
          continue;
        }

        const DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(posEleDiff);
        if (!digiPadPos.isValid()) {
          continue;
        }

        /// Remove digits the end up outside the currently produced sector
        if (digiPadPos.getCRU().sector() != sector) {
          continue;
        }

        const int nElectronsGEM = gemAmplification.getStackAmplification();
        if (nElectronsGEM == 0) {
          continue;
        }

        /// Loop over all individual pads with signal due to pad response function
        /// Currently the PRF is not applied yet due to some problems with the mapper
        /// which results in most of the cases in a normalized pad response = 0
        /// \todo Problems of the mapper to be fixed
        /// \todo Mapper should provide a functionality which finds the adjacent pads of a given pad
        // for(int ipad = -2; ipad<3; ++ipad) {
        //   for(int irow = -2; irow<3; ++irow) {
        //     PadPos padPos(digiPadPos.getPadPos().getRow() + irow, digiPadPos.getPadPos().getPad() + ipad);
        //     DigitPos digiPos(digiPadPos.getCRU(), padPos);

        DigitPos digiPos = digiPadPos;
        if (!digiPos.isValid()) {
          continue;
        }
        // const float normalizedPadResponse = padResponse.getPadResponse(posEleDiff, digiPos);

        const float normalizedPadResponse = 1.f;
        if (normalizedPadResponse <= 0) {
          continue;
        }

        const int pad = digiPos.getPadPos().getPad();
        const int row = digiPos.getPadPos().getRow();
        const GlobalPadNumber globalPad =
          mapper.getPadNumberInROC(PadROCPos(digiPadPos.getCRU().roc(), PadPos(row, pad)));

        const float ADCsignal = SAMPAProcessing::getADCvalue(nElectronsGEM * normalizedPadResponse);
        SAMPAProcessing::getShapedSignal(ADCsignal, absoluteTime, signalArray);
        for (float i = 0; i < nShapedPoints; ++i) {
          const float time = absoluteTime + i * eleParam.getZBinWidth();
          mDigitContainer->addDigit(eventID, MCTrackID, digiPos.getCRU(), SAMPAProcessing::getTimeBinFromTime(time),
                                    globalPad, signalArray[i]);
        }

        // }
        // }
        /// end of loop over prf
      }
      /// end of loop over electrons
    }
  }
  /// end of loop over points

  return mDigitContainer;
}
