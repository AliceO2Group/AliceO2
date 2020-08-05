// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// Class to provide PID response
/// \file   PIDResponse.h
/// \author Nicolo' Jacazio
///

#ifndef O2_FRAMEWORK_PIDRESPONSE_H_
#define O2_FRAMEWORK_PIDRESPONSE_H_

// O2 includes
#include "Framework/ASoA.h"
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/PID.h"
#include "PID/PIDTOF.h"
#include "PID/PIDTPC.h"

namespace o2::aod
{

namespace pidTOFbeta
{
DECLARE_SOA_COLUMN(Beta, beta, float);
DECLARE_SOA_COLUMN(BetaError, betaerror, float);
//
DECLARE_SOA_COLUMN(ExpBetaEl, expbetael, float);
DECLARE_SOA_COLUMN(ExpBetaElError, expbetaelerror, float);
//
DECLARE_SOA_COLUMN(SeparationBetaEl, separationbetael, float);
DECLARE_SOA_DYNAMIC_COLUMN(DiffBetaEl, diffbetael, [](float beta, float expbetael) -> float { return beta - expbetael; });
} // namespace pidTOFbeta

namespace pidTOF
{
// Expected times
DECLARE_SOA_COLUMN(TOFExpSignalEl, tofExpSignalEl, float);
DECLARE_SOA_COLUMN(TOFExpSignalMu, tofExpSignalMu, float);
DECLARE_SOA_COLUMN(TOFExpSignalPi, tofExpSignalPi, float);
DECLARE_SOA_COLUMN(TOFExpSignalKa, tofExpSignalKa, float);
DECLARE_SOA_COLUMN(TOFExpSignalPr, tofExpSignalPr, float);
DECLARE_SOA_COLUMN(TOFExpSignalDe, tofExpSignalDe, float);
DECLARE_SOA_COLUMN(TOFExpSignalTr, tofExpSignalTr, float);
DECLARE_SOA_COLUMN(TOFExpSignalHe, tofExpSignalHe, float);
DECLARE_SOA_COLUMN(TOFExpSignalAl, tofExpSignalAl, float);
// Expected sigma
DECLARE_SOA_COLUMN(TOFExpSigmaEl, tofExpSigmaEl, float);
DECLARE_SOA_COLUMN(TOFExpSigmaMu, tofExpSigmaMu, float);
DECLARE_SOA_COLUMN(TOFExpSigmaPi, tofExpSigmaPi, float);
DECLARE_SOA_COLUMN(TOFExpSigmaKa, tofExpSigmaKa, float);
DECLARE_SOA_COLUMN(TOFExpSigmaPr, tofExpSigmaPr, float);
DECLARE_SOA_COLUMN(TOFExpSigmaDe, tofExpSigmaDe, float);
DECLARE_SOA_COLUMN(TOFExpSigmaTr, tofExpSigmaTr, float);
DECLARE_SOA_COLUMN(TOFExpSigmaHe, tofExpSigmaHe, float);
DECLARE_SOA_COLUMN(TOFExpSigmaAl, tofExpSigmaAl, float);
// NSigma
DECLARE_SOA_COLUMN(TOFNSigmaEl, tofNSigmaEl, float);
DECLARE_SOA_COLUMN(TOFNSigmaMu, tofNSigmaMu, float);
DECLARE_SOA_COLUMN(TOFNSigmaPi, tofNSigmaPi, float);
DECLARE_SOA_COLUMN(TOFNSigmaKa, tofNSigmaKa, float);
DECLARE_SOA_COLUMN(TOFNSigmaPr, tofNSigmaPr, float);
DECLARE_SOA_COLUMN(TOFNSigmaDe, tofNSigmaDe, float);
DECLARE_SOA_COLUMN(TOFNSigmaTr, tofNSigmaTr, float);
DECLARE_SOA_COLUMN(TOFNSigmaHe, tofNSigmaHe, float);
DECLARE_SOA_COLUMN(TOFNSigmaAl, tofNSigmaAl, float);
} // namespace pidTOF

using namespace pidTOFbeta;
DECLARE_SOA_TABLE(pidRespTOFbeta, "AOD", "pidRespTOFbeta",
                  Beta, BetaError,
                  ExpBetaEl, ExpBetaElError,
                  SeparationBetaEl,
                  DiffBetaEl<Beta, ExpBetaEl>);
using namespace pidTOF;
DECLARE_SOA_TABLE(pidRespTOF, "AOD", "pidRespTOF",
                  TOFExpSignalEl, TOFExpSignalMu, TOFExpSignalPi, TOFExpSignalKa, TOFExpSignalPr, TOFExpSignalDe, TOFExpSignalTr, TOFExpSignalHe, TOFExpSignalAl,
                  TOFExpSigmaEl, TOFExpSigmaMu, TOFExpSigmaPi, TOFExpSigmaKa, TOFExpSigmaPr, TOFExpSigmaDe, TOFExpSigmaTr, TOFExpSigmaHe, TOFExpSigmaAl,
                  TOFNSigmaEl, TOFNSigmaMu, TOFNSigmaPi, TOFNSigmaKa, TOFNSigmaPr, TOFNSigmaDe, TOFNSigmaTr, TOFNSigmaHe, TOFNSigmaAl);

namespace pidTPC
{
// Expected signals
DECLARE_SOA_COLUMN(TPCExpSignalEl, tpcExpSignalEl, float);
DECLARE_SOA_COLUMN(TPCExpSignalMu, tpcExpSignalMu, float);
DECLARE_SOA_COLUMN(TPCExpSignalPi, tpcExpSignalPi, float);
DECLARE_SOA_COLUMN(TPCExpSignalKa, tpcExpSignalKa, float);
DECLARE_SOA_COLUMN(TPCExpSignalPr, tpcExpSignalPr, float);
DECLARE_SOA_COLUMN(TPCExpSignalDe, tpcExpSignalDe, float);
DECLARE_SOA_COLUMN(TPCExpSignalTr, tpcExpSignalTr, float);
DECLARE_SOA_COLUMN(TPCExpSignalHe, tpcExpSignalHe, float);
DECLARE_SOA_COLUMN(TPCExpSignalAl, tpcExpSignalAl, float);
// Expected sigma
DECLARE_SOA_COLUMN(TPCExpSigmaEl, tpcExpSigmaEl, float);
DECLARE_SOA_COLUMN(TPCExpSigmaMu, tpcExpSigmaMu, float);
DECLARE_SOA_COLUMN(TPCExpSigmaPi, tpcExpSigmaPi, float);
DECLARE_SOA_COLUMN(TPCExpSigmaKa, tpcExpSigmaKa, float);
DECLARE_SOA_COLUMN(TPCExpSigmaPr, tpcExpSigmaPr, float);
DECLARE_SOA_COLUMN(TPCExpSigmaDe, tpcExpSigmaDe, float);
DECLARE_SOA_COLUMN(TPCExpSigmaTr, tpcExpSigmaTr, float);
DECLARE_SOA_COLUMN(TPCExpSigmaHe, tpcExpSigmaHe, float);
DECLARE_SOA_COLUMN(TPCExpSigmaAl, tpcExpSigmaAl, float);
// NSigma
DECLARE_SOA_COLUMN(TPCNSigmaEl, tpcNSigmaEl, float);
DECLARE_SOA_COLUMN(TPCNSigmaMu, tpcNSigmaMu, float);
DECLARE_SOA_COLUMN(TPCNSigmaPi, tpcNSigmaPi, float);
DECLARE_SOA_COLUMN(TPCNSigmaKa, tpcNSigmaKa, float);
DECLARE_SOA_COLUMN(TPCNSigmaPr, tpcNSigmaPr, float);
DECLARE_SOA_COLUMN(TPCNSigmaDe, tpcNSigmaDe, float);
DECLARE_SOA_COLUMN(TPCNSigmaTr, tpcNSigmaTr, float);
DECLARE_SOA_COLUMN(TPCNSigmaHe, tpcNSigmaHe, float);
DECLARE_SOA_COLUMN(TPCNSigmaAl, tpcNSigmaAl, float);
} // namespace pidTPC

using namespace pidTPC;
DECLARE_SOA_TABLE(pidRespTPC, "AOD", "pidRespTPC",
                  TPCExpSignalEl, TPCExpSignalMu, TPCExpSignalPi, TPCExpSignalKa, TPCExpSignalPr, TPCExpSignalDe, TPCExpSignalTr, TPCExpSignalHe, TPCExpSignalAl,
                  TPCExpSigmaEl, TPCExpSigmaMu, TPCExpSigmaPi, TPCExpSigmaKa, TPCExpSigmaPr, TPCExpSigmaDe, TPCExpSigmaTr, TPCExpSigmaHe, TPCExpSigmaAl,
                  TPCNSigmaEl, TPCNSigmaMu, TPCNSigmaPi, TPCNSigmaKa, TPCNSigmaPr, TPCNSigmaDe, TPCNSigmaTr, TPCNSigmaHe, TPCNSigmaAl);

} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::pid;
using namespace o2::framework::expressions;
using namespace o2::track;

struct pidTOFTask {
  Produces<aod::pidRespTOF> tofpid;
  Produces<aod::pidRespTOFbeta> tofpidbeta;

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    LOGF(info, "Tracks for collision: %d", tracks.size());
    tof::EventTime evt = tof::EventTime();
    evt.SetEvTime(0, collision.collisionTime());
    evt.SetEvTimeReso(0, collision.collisionTimeRes());
    evt.SetEvTimeMask(0, collision.collisionTimeMask());
    tof::Response resp = tof::Response();
    resp.SetEventTime(evt);
    for (auto const& i : tracks) {
      resp.UpdateTrack(i.p(), i.tofExpMom() / tof::Response::kCSPEED, i.length(), i.tofSignal());
      tofpidbeta(resp.GetBeta(),
                 resp.GetBetaExpectedSigma(),
                 resp.GetExpectedBeta(PID::Electron),
                 resp.GetBetaExpectedSigma(),
                 resp.GetBetaNumberOfSigmas(PID::Electron));
      tofpid(
        resp.GetExpectedSignal(PID::Electron),
        resp.GetExpectedSignal(PID::Muon),
        resp.GetExpectedSignal(PID::Pion),
        resp.GetExpectedSignal(PID::Kaon),
        resp.GetExpectedSignal(PID::Proton),
        resp.GetExpectedSignal(PID::Deuteron),
        resp.GetExpectedSignal(PID::Triton),
        resp.GetExpectedSignal(PID::Helium3),
        resp.GetExpectedSignal(PID::Alpha),
        resp.GetExpectedSigma(PID::Electron),
        resp.GetExpectedSigma(PID::Muon),
        resp.GetExpectedSigma(PID::Pion),
        resp.GetExpectedSigma(PID::Kaon),
        resp.GetExpectedSigma(PID::Proton),
        resp.GetExpectedSigma(PID::Deuteron),
        resp.GetExpectedSigma(PID::Triton),
        resp.GetExpectedSigma(PID::Helium3),
        resp.GetExpectedSigma(PID::Alpha),
        resp.GetNumberOfSigmas(PID::Electron),
        resp.GetNumberOfSigmas(PID::Muon),
        resp.GetNumberOfSigmas(PID::Pion),
        resp.GetNumberOfSigmas(PID::Kaon),
        resp.GetNumberOfSigmas(PID::Proton),
        resp.GetNumberOfSigmas(PID::Deuteron),
        resp.GetNumberOfSigmas(PID::Triton),
        resp.GetNumberOfSigmas(PID::Helium3),
        resp.GetNumberOfSigmas(PID::Alpha));
    }
  }
};

struct pidTPCTask {
  Produces<aod::pidRespTPC> tpcpid;

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    tpc::Response resp = tpc::Response();
    float bbparams[5] = {0.0320981, 19.9768, 2.52666e-16, 2.72123, 6.08092};
    resp.mParam.mBetheBloch.mParameters.Set(bbparams);
    float resoparams[2] = {0.07, 0.0};
    resp.mParam.mRelResolution.mParameters.Set(resoparams);
    for (auto const& i : tracks) {
      resp.UpdateTrack(i.p(), i.tpcSignal(), i.tpcNClsShared());
      tpcpid(
        resp.GetExpectedSignal(PID::Electron),
        resp.GetExpectedSignal(PID::Muon),
        resp.GetExpectedSignal(PID::Pion),
        resp.GetExpectedSignal(PID::Kaon),
        resp.GetExpectedSignal(PID::Proton),
        resp.GetExpectedSignal(PID::Deuteron),
        resp.GetExpectedSignal(PID::Triton),
        resp.GetExpectedSignal(PID::Helium3),
        resp.GetExpectedSignal(PID::Alpha),
        resp.GetExpectedSigma(PID::Electron),
        resp.GetExpectedSigma(PID::Muon),
        resp.GetExpectedSigma(PID::Pion),
        resp.GetExpectedSigma(PID::Kaon),
        resp.GetExpectedSigma(PID::Proton),
        resp.GetExpectedSigma(PID::Deuteron),
        resp.GetExpectedSigma(PID::Triton),
        resp.GetExpectedSigma(PID::Helium3),
        resp.GetExpectedSigma(PID::Alpha),
        resp.GetNumberOfSigmas(PID::Electron),
        resp.GetNumberOfSigmas(PID::Muon),
        resp.GetNumberOfSigmas(PID::Pion),
        resp.GetNumberOfSigmas(PID::Kaon),
        resp.GetNumberOfSigmas(PID::Proton),
        resp.GetNumberOfSigmas(PID::Deuteron),
        resp.GetNumberOfSigmas(PID::Triton),
        resp.GetNumberOfSigmas(PID::Helium3),
        resp.GetNumberOfSigmas(PID::Alpha));
    }
  }
};

#endif // O2_FRAMEWORK_PIDRESPONSE_H_
