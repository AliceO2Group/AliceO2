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
  if (!mIsContinuous) {
    eventTime = 0.f;
  }

  for (auto& inputgroup : hits) {
    ProcessHitGroup(inputgroup, sector, eventTime, eventID);
  }
  return mDigitContainer;
}

DigitContainer* Digitizer::Process2(const Sector& sector, const std::vector<std::vector<o2::TPC::HitGroup>*>& hits,
                                    const std::vector<o2::TPC::TPCHitGroupID>& hitids,
                                    const o2::steer::RunContext& context)
{
  const auto& interactRecords = context.getEventRecords();

  for (auto& id : hitids) {
    const auto hitvector = hits[id.storeindex];
    auto& group = (*hitvector)[id.groupID];
    auto& MCrecord = interactRecords[id.entry];
    ProcessHitGroup(group, sector, MCrecord.timeNS * 0.001f, id.entry, id.sourceID);
  }

  return mDigitContainer;
}

void Digitizer::ProcessHitGroup(const HitGroup& inputgroup, const Sector& sector, const float eventTime,
                                const int eventID, const int sourceID)
{
  const static Mapper& mapper = Mapper::instance();
  const static ParameterDetector& detParam = ParameterDetector::defaultInstance();
  const static ParameterElectronics& eleParam = ParameterElectronics::defaultInstance();

  static GEMAmplification& gemAmplification = GEMAmplification::instance();
  gemAmplification.updateParameters();
  static ElectronTransport& electronTransport = ElectronTransport::instance();
  electronTransport.updateParameters();
  static SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  sampaProcessing.updateParameters();

  const int nShapedPoints = eleParam.getNShapedPoints();
  static std::vector<float> signalArray;
  signalArray.resize(nShapedPoints);

  const int MCTrackID = inputgroup.GetTrackID();
  for (size_t hitindex = 0; hitindex < inputgroup.getSize(); ++hitindex) {
    const auto& eh = inputgroup.getHit(hitindex);

    const GlobalPosition3D posEle(eh.GetX(), eh.GetY(), eh.GetZ());

    /// Remove electrons that end up more than three sigma of the hit's average diffusion away from the current sector
    /// boundary
    if (electronTransport.isCompletelyOutOfSectorCoarseElectronDrift(posEle, sector)) {
      continue;
    }

    /// The energy loss stored corresponds to nElectrons
    const int nPrimaryElectrons = static_cast<int>(eh.GetEnergyLoss());

    /// Loop over electrons
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
      if (std::abs(posEleDiff.Z()) > detParam.getTPClength()) {
        continue;
      }

      /// Compute digit position and check for validity
      const DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(posEleDiff);
      if (!digiPadPos.isValid()) {
        continue;
      }

      /// Remove digits the end up outside the currently produced sector
      if (digiPadPos.getCRU().sector() != sector) {
        continue;
      }

      /// Electron amplification
      const int nElectronsGEM = gemAmplification.getStackAmplification(digiPadPos.getCRU(), digiPadPos.getPadPos());
      if (nElectronsGEM == 0) {
        continue;
      }

      const GlobalPadNumber globalPad = mapper.globalPadNumber(digiPadPos.getGlobalPadPos());
      const float ADCsignal = sampaProcessing.getADCvalue(static_cast<float>(nElectronsGEM));
      sampaProcessing.getShapedSignal(ADCsignal, absoluteTime, signalArray);
      for (float i = 0; i < nShapedPoints; ++i) {
        const float time = absoluteTime + i * eleParam.getZBinWidth();
        const MCCompLabel label(MCTrackID, eventID, sourceID);
        mDigitContainer->addDigit(label, digiPadPos.getCRU(), sampaProcessing.getTimeBinFromTime(time), globalPad,
                                  signalArray[i]);
      }
    }
    /// end of loop over electrons
  }
}
