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
/// \file   PIDResponse.h
/// \author Nicolo' Jacazio
///

#ifndef O2_FRAMEWORK_PIDRESPONSE_H_
#define O2_FRAMEWORK_PIDRESPONSE_H_

// ROOT includes
#include "TMath.h"

// O2 includes
#include "Framework/ASoA.h"
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "ReconstructionDataFormats/Track.h"
#include "PID/PIDTOF.h"

namespace o2::aod
{

namespace pidTOF
{
DECLARE_SOA_COLUMN(Beta, beta, float);
DECLARE_SOA_COLUMN(BetaError, betaerror, float);
//
DECLARE_SOA_COLUMN(ExpBetaEl, expbetael, float);
DECLARE_SOA_COLUMN(ExpBetaElError, expbetaelerror, float);
//
DECLARE_SOA_COLUMN(SeparationBetaEl, separationbetael, float);
DECLARE_SOA_DYNAMIC_COLUMN(DiffBetaEl, diffbetael, [](float beta, float expbetael) -> float { return beta - expbetael; });
//
DECLARE_SOA_COLUMN(ExpSigmaPi, expsigmaPi, float);
DECLARE_SOA_COLUMN(ExpSigmaKa, expsigmaKa, float);
DECLARE_SOA_COLUMN(ExpSigmaPr, expsigmaPr, float);
//
DECLARE_SOA_COLUMN(NSigmaPi, nsigmaPi, float);
DECLARE_SOA_COLUMN(NSigmaKa, nsigmaKa, float);
DECLARE_SOA_COLUMN(NSigmaPr, nsigmaPr, float);
//
DECLARE_SOA_COLUMN(ExppEl, exppEl, float);
DECLARE_SOA_COLUMN(ExppMu, exppMu, float);
DECLARE_SOA_COLUMN(ExppPi, exppPi, float);
DECLARE_SOA_COLUMN(ExppKa, exppKa, float);
DECLARE_SOA_COLUMN(ExppPr, exppPr, float);
DECLARE_SOA_COLUMN(ExppDe, exppDe, float);
//
DECLARE_SOA_COLUMN(CExpTimeEl_El, cexpTimeEl_El, float);
DECLARE_SOA_COLUMN(CExpTimeMu_El, cexpTimeMu_El, float);
DECLARE_SOA_COLUMN(CExpTimePi_El, cexpTimePi_El, float);
DECLARE_SOA_COLUMN(CExpTimeKa_El, cexpTimeKa_El, float);
DECLARE_SOA_COLUMN(CExpTimePr_El, cexpTimePr_El, float);
DECLARE_SOA_COLUMN(CExpTimeDe_El, cexpTimeDe_El, float);
//
// DECLARE_SOA_COLUMN(CExpTimeEl_Mu, cexpTimeEl_Mu, float);
// DECLARE_SOA_COLUMN(CExpTimeMu_Mu, cexpTimeMu_Mu, float);
// DECLARE_SOA_COLUMN(CExpTimePi_Mu, cexpTimePi_Mu, float);
// DECLARE_SOA_COLUMN(CExpTimeKa_Mu, cexpTimeKa_Mu, float);
// DECLARE_SOA_COLUMN(CExpTimePr_Mu, cexpTimePr_Mu, float);
// DECLARE_SOA_COLUMN(CExpTimeDe_Mu, cexpTimeDe_Mu, float);
// //
// DECLARE_SOA_COLUMN(CExpTimeEl_Pi, cexpTimeEl_Pi, float);
// DECLARE_SOA_COLUMN(CExpTimeMu_Pi, cexpTimeMu_Pi, float);
// DECLARE_SOA_COLUMN(CExpTimePi_Pi, cexpTimePi_Pi, float);
// DECLARE_SOA_COLUMN(CExpTimeKa_Pi, cexpTimeKa_Pi, float);
// DECLARE_SOA_COLUMN(CExpTimePr_Pi, cexpTimePr_Pi, float);
// DECLARE_SOA_COLUMN(CExpTimeDe_Pi, cexpTimeDe_Pi, float);
// //
// DECLARE_SOA_COLUMN(CExpTimeEl_Ka, cexpTimeEl_Ka, float);
// DECLARE_SOA_COLUMN(CExpTimeMu_Ka, cexpTimeMu_Ka, float);
// DECLARE_SOA_COLUMN(CExpTimePi_Ka, cexpTimePi_Ka, float);
// DECLARE_SOA_COLUMN(CExpTimeKa_Ka, cexpTimeKa_Ka, float);
// DECLARE_SOA_COLUMN(CExpTimePr_Ka, cexpTimePr_Ka, float);
// DECLARE_SOA_COLUMN(CExpTimeDe_Ka, cexpTimeDe_Ka, float);
// //
// DECLARE_SOA_COLUMN(CExpTimeEl_Pr, cexpTimeEl_Pr, float);
// DECLARE_SOA_COLUMN(CExpTimeMu_Pr, cexpTimeMu_Pr, float);
// DECLARE_SOA_COLUMN(CExpTimePi_Pr, cexpTimePi_Pr, float);
// DECLARE_SOA_COLUMN(CExpTimeKa_Pr, cexpTimeKa_Pr, float);
// DECLARE_SOA_COLUMN(CExpTimePr_Pr, cexpTimePr_Pr, float);
// DECLARE_SOA_COLUMN(CExpTimeDe_Pr, cexpTimeDe_Pr, float);
// //
// DECLARE_SOA_COLUMN(CExpTimeEl_De, cexpTimeEl_De, float);
// DECLARE_SOA_COLUMN(CExpTimeMu_De, cexpTimeMu_De, float);
// DECLARE_SOA_COLUMN(CExpTimePi_De, cexpTimePi_De, float);
// DECLARE_SOA_COLUMN(CExpTimeKa_De, cexpTimeKa_De, float);
// DECLARE_SOA_COLUMN(CExpTimePr_De, cexpTimePr_De, float);
// DECLARE_SOA_COLUMN(CExpTimeDe_De, cexpTimeDe_De, float);
//
} // namespace pidTOF
using namespace pidTOF;
DECLARE_SOA_TABLE(pidRespTOF, "AOD", "pidRespTOF",
                  Beta,
                  BetaError,
                  ExpBetaEl,
                  ExpBetaElError,
                  SeparationBetaEl,
                  DiffBetaEl<Beta, ExpBetaEl>,
                  ExpSigmaPi,
                  ExpSigmaKa,
                  ExpSigmaPr,
                  NSigmaPi,
                  NSigmaKa,
                  NSigmaPr
                  // ExppEl,
                  // ExppMu,
                  // ExppPi,
                  // ExppKa,
                  // ExppPr,
                  // ExppDe
                  // CExpTimeEl_El,
                  // CExpTimeMu_El,
                  // CExpTimePi_El,
                  // CExpTimeKa_El,
                  // CExpTimePr_El,
                  // CExpTimeDe_El
                  // //
                  // CExpTimeEl_Mu,
                  // CExpTimeMu_Mu,
                  // CExpTimePi_Mu,
                  // CExpTimeKa_Mu,
                  // CExpTimePr_Mu,
                  // CExpTimeDe_Mu,
                  // //
                  // CExpTimeEl_Pi,
                  // CExpTimeMu_Pi,
                  // CExpTimePi_Pi,
                  // CExpTimeKa_Pi,
                  // CExpTimePr_Pi,
                  // CExpTimeDe_Pi,
                  // //
                  // CExpTimeEl_Ka,
                  // CExpTimeMu_Ka,
                  // CExpTimePi_Ka,
                  // CExpTimeKa_Ka,
                  // CExpTimePr_Ka,
                  // CExpTimeDe_Ka,
                  // //
                  // CExpTimeEl_Pr,
                  // CExpTimeMu_Pr,
                  // CExpTimePi_Pr,
                  // CExpTimeKa_Pr,
                  // CExpTimePr_Pr,
                  // CExpTimeDe_Pr,
                  // //
                  // CExpTimeEl_De,
                  // CExpTimeMu_De,
                  // CExpTimePi_De,
                  // CExpTimeKa_De,
                  // CExpTimePr_De,
                  // CExpTimeDe_De
                  // //
);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;
using namespace o2::pid::tof;
using namespace o2::framework::expressions;

struct pidTOFTask {
  Produces<aod::pidRespTOF> tofpid;

  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra> const& tracks)
  {
    LOGF(info, "Tracks for collision: %d", tracks.size());
    Response resp = pid::tof::Response();
    for (auto i : tracks) {
      // float EVTIME = collision.collisionTime0();
      float EVTIME = collision.collisionTime();
      float BETA = beta(i.length(), i.tofSignal(), EVTIME);
      float BETAERROR = betaerror(i.length(), i.tofSignal(), EVTIME);
      float EXPBETAEL = expbeta(p(i.eta(), i.signed1Pt()), kElectronMass);
      float EXPBETAELERROR = 0;
      //  resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes0(), kPionMass),
      //  resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes0(), kKaonMass),
      //  resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes0(), kProtonMass),
      //
      //  0,0,0
      //  resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPi(), collision.collisionTime0(), collision.collisionTimeRes0(), kPionMass),
      //  resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpKa(), collision.collisionTime0(), collision.collisionTimeRes0(), kKaonMass),
      //  resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPr(), collision.collisionTime0(), collision.collisionTimeRes0(), kProtonMass)
      tofpid(BETA,
             BETAERROR,
             EXPBETAEL,
             EXPBETAELERROR,
             BETA > 0 ? (BETA - EXPBETAEL) / sqrt(BETAERROR * BETAERROR + EXPBETAELERROR * EXPBETAELERROR) : -999,
             resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), kPionMass),
             resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), kKaonMass),
             resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), kProtonMass),
             resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPi(), collision.collisionTime(), collision.collisionTimeRes(), kPionMass),
             resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpKa(), collision.collisionTime(), collision.collisionTimeRes(), kKaonMass),
             resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPr(), collision.collisionTime(), collision.collisionTimeRes(), kProtonMass)
            //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpEl(), i.length(), kElectronMass), i.length(), kElectronMass),
            //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpEl(), i.length(), kElectronMass), i.length(), kMuonMass),
            //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpEl(), i.length(), kElectronMass), i.length(), kPionMass),
            //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpEl(), i.length(), kElectronMass), i.length(), kKaonMass),
            //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpEl(), i.length(), kElectronMass), i.length(), kProtonMass),
            //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpEl(), i.length(), kElectronMass), i.length(), kDeuteronMass)
             //  //
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpMu(), i.length(), kMuonMass), i.length(), kElectronMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpMu(), i.length(), kMuonMass), i.length(), kMuonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpMu(), i.length(), kMuonMass), i.length(), kPionMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpMu(), i.length(), kMuonMass), i.length(), kKaonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpMu(), i.length(), kMuonMass), i.length(), kProtonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpMu(), i.length(), kMuonMass), i.length(), kDeuteronMass),
             //  //
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPi(), i.length(), kPionMass), i.length(), kElectronMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPi(), i.length(), kPionMass), i.length(), kMuonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPi(), i.length(), kPionMass), i.length(), kPionMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPi(), i.length(), kPionMass), i.length(), kKaonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPi(), i.length(), kPionMass), i.length(), kProtonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPi(), i.length(), kPionMass), i.length(), kDeuteronMass),
             //  //
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpKa(), i.length(), kKaonMass), i.length(), kElectronMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpKa(), i.length(), kKaonMass), i.length(), kMuonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpKa(), i.length(), kKaonMass), i.length(), kPionMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpKa(), i.length(), kKaonMass), i.length(), kKaonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpKa(), i.length(), kKaonMass), i.length(), kProtonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpKa(), i.length(), kKaonMass), i.length(), kDeuteronMass),
             //  //
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPr(), i.length(), kProtonMass), i.length(), kElectronMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPr(), i.length(), kProtonMass), i.length(), kMuonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPr(), i.length(), kProtonMass), i.length(), kPionMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPr(), i.length(), kProtonMass), i.length(), kKaonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPr(), i.length(), kProtonMass), i.length(), kProtonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpPr(), i.length(), kProtonMass), i.length(), kDeuteronMass),
             //  //
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpDe(), i.length(), kDeuteronMass), i.length(), kElectronMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpDe(), i.length(), kDeuteronMass), i.length(), kMuonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpDe(), i.length(), kDeuteronMass), i.length(), kPionMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpDe(), i.length(), kDeuteronMass), i.length(), kKaonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpDe(), i.length(), kDeuteronMass), i.length(), kProtonMass),
             //  ComputeTOFExpTime(ComputeExpectedMomentum(i.tofExpDe(), i.length(), kDeuteronMass), i.length(), kDeuteronMass)
             //  //
      );
    }
  }
};

#endif // O2_FRAMEWORK_PIDRESPONSE_H_
