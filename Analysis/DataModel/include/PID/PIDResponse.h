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
using namespace o2::pid::tof;
using namespace o2::framework::expressions;
using namespace o2::track;

struct pidTOFTask {
  Produces<aod::pidRespTOF> tofpid;
  Produces<aod::pidRespTOFbeta> tofpidbeta;

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    LOGF(info, "Tracks for collision: %d", tracks.size());
    Response resp = pid::tof::Response();
    for (auto i : tracks) {
      resp.InitResponse(i.p(), i.tofExpMom(), i.length(), i.tofSignal());
      // float EVTIME = collision.collisionTime0();
      float EVTIME = collision.collisionTime();
      float EXPBETAEL = expbeta(p(i.eta(), i.signed1Pt()), PID::getMass(PID::Electron));
      float EXPBETAELERROR = 0;
      tofpidbeta(resp.GetBeta(),
                 resp.GetBetaExpectedSigma(),
                 EXPBETAEL,
                 EXPBETAELERROR,
                 resp.GetBeta() > 0 ? (resp.GetBeta() - EXPBETAEL) / sqrt(resp.GetBetaExpectedSigma() * resp.GetBetaExpectedSigma() + EXPBETAELERROR * EXPBETAELERROR) : -999);
      tofpid(
        ComputeTOFExpTime(i.tofExpMom(), i.length(), PID::getMass(PID::Electron)),
        ComputeTOFExpTime(i.tofExpMom(), i.length(), PID::getMass(PID::Muon)),
        ComputeTOFExpTime(i.tofExpMom(), i.length(), PID::getMass(PID::Pion)),
        ComputeTOFExpTime(i.tofExpMom(), i.length(), PID::getMass(PID::Kaon)),
        ComputeTOFExpTime(i.tofExpMom(), i.length(), PID::getMass(PID::Proton)),
        ComputeTOFExpTime(i.tofExpMom(), i.length(), PID::getMass(PID::Deuteron)),
        ComputeTOFExpTime(i.tofExpMom(), i.length(), PID::getMass(PID::Triton)),
        ComputeTOFExpTime(i.tofExpMom(), i.length(), PID::getMass(PID::Helium3)),
        ComputeTOFExpTime(i.tofExpMom(), i.length(), PID::getMass(PID::Alpha)),
        resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), PID::getMass(PID::Electron)),
        resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), PID::getMass(PID::Muon)),
        resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), PID::getMass(PID::Pion)),
        resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), PID::getMass(PID::Kaon)),
        resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), PID::getMass(PID::Proton)),
        resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), PID::getMass(PID::Deuteron)),
        resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), PID::getMass(PID::Triton)),
        resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), PID::getMass(PID::Helium3)),
        resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), PID::getMass(PID::Alpha)),
        resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPi(), collision.collisionTime(), collision.collisionTimeRes(), PID::getMass(PID::Electron)),
        resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPi(), collision.collisionTime(), collision.collisionTimeRes(), PID::getMass(PID::Muon)),
        resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPi(), collision.collisionTime(), collision.collisionTimeRes(), PID::getMass(PID::Pion)),
        resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpKa(), collision.collisionTime(), collision.collisionTimeRes(), PID::getMass(PID::Kaon)),
        resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPr(), collision.collisionTime(), collision.collisionTimeRes(), PID::getMass(PID::Proton)),
        resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPi(), collision.collisionTime(), collision.collisionTimeRes(), PID::getMass(PID::Deuteron)),
        resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPi(), collision.collisionTime(), collision.collisionTimeRes(), PID::getMass(PID::Triton)),
        resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPi(), collision.collisionTime(), collision.collisionTimeRes(), PID::getMass(PID::Helium3)),
        resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPi(), collision.collisionTime(), collision.collisionTimeRes(), PID::getMass(PID::Alpha)));
    }
  }
};

#endif // O2_FRAMEWORK_PIDRESPONSE_H_
