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
#include "PID/PIDTOF.h"
#include "ReconstructionDataFormats/PID.h"

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
DECLARE_SOA_COLUMN(ExpTimeEl, expTimeEl, float);
DECLARE_SOA_COLUMN(ExpTimeMu, expTimeMu, float);
DECLARE_SOA_COLUMN(ExpTimePi, expTimePi, float);
DECLARE_SOA_COLUMN(ExpTimeKa, expTimeKa, float);
DECLARE_SOA_COLUMN(ExpTimePr, expTimePr, float);
DECLARE_SOA_COLUMN(ExpTimeDe, expTimeDe, float);
DECLARE_SOA_COLUMN(ExpTimeTr, expTimeTr, float);
DECLARE_SOA_COLUMN(ExpTimeHe, expTimeHe, float);
DECLARE_SOA_COLUMN(ExpTimeAl, expTimeAl, float);
// Expected sigma
DECLARE_SOA_COLUMN(ExpSigmaEl, expSigmaEl, float);
DECLARE_SOA_COLUMN(ExpSigmaMu, expSigmaMu, float);
DECLARE_SOA_COLUMN(ExpSigmaPi, expSigmaPi, float);
DECLARE_SOA_COLUMN(ExpSigmaKa, expSigmaKa, float);
DECLARE_SOA_COLUMN(ExpSigmaPr, expSigmaPr, float);
DECLARE_SOA_COLUMN(ExpSigmaDe, expSigmaDe, float);
DECLARE_SOA_COLUMN(ExpSigmaTr, expSigmaTr, float);
DECLARE_SOA_COLUMN(ExpSigmaHe, expSigmaHe, float);
DECLARE_SOA_COLUMN(ExpSigmaAl, expSigmaAl, float);
// NSigma
DECLARE_SOA_COLUMN(NSigmaEl, nSigmaEl, float);
DECLARE_SOA_COLUMN(NSigmaMu, nSigmaMu, float);
DECLARE_SOA_COLUMN(NSigmaPi, nSigmaPi, float);
DECLARE_SOA_COLUMN(NSigmaKa, nSigmaKa, float);
DECLARE_SOA_COLUMN(NSigmaPr, nSigmaPr, float);
DECLARE_SOA_COLUMN(NSigmaDe, nSigmaDe, float);
DECLARE_SOA_COLUMN(NSigmaTr, nSigmaTr, float);
DECLARE_SOA_COLUMN(NSigmaHe, nSigmaHe, float);
DECLARE_SOA_COLUMN(NSigmaAl, nSigmaAl, float);
} // namespace pidTOF

using namespace pidTOFbeta;
DECLARE_SOA_TABLE(pidRespTOFbeta, "AOD", "pidRespTOFbeta",
                  Beta, BetaError,
                  ExpBetaEl, ExpBetaElError,
                  SeparationBetaEl,
                  DiffBetaEl<Beta, ExpBetaEl>);
using namespace pidTOF;
DECLARE_SOA_TABLE(pidRespTOF, "AOD", "pidRespTOF",
                  ExpTimeEl, ExpTimeMu, ExpTimePi, ExpTimeKa, ExpTimePr, ExpTimeDe, ExpTimeTr, ExpTimeHe, ExpTimeAl,
                  ExpSigmaEl, ExpSigmaMu, ExpSigmaPi, ExpSigmaKa, ExpSigmaPr, ExpSigmaDe, ExpSigmaTr, ExpSigmaHe, ExpSigmaAl,
                  NSigmaEl, NSigmaMu, NSigmaPi, NSigmaKa, NSigmaPr, NSigmaDe, NSigmaTr, NSigmaHe, NSigmaAl);

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
    for (auto i : tracks) {
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

#endif // O2_FRAMEWORK_PIDRESPONSE_H_
