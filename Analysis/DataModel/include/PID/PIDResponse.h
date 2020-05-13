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

// namespace pidTOF
// {
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
// } // namespace pidTOF
// using namespace pidTOF;
DECLARE_SOA_TABLE(pidRespTOF, "AOD", "pidRespTOF",
                  Beta,
                  BetaError,
                  //
                  ExpBetaEl,
                  ExpBetaElError,
                  SeparationBetaEl,
                  DiffBetaEl<Beta, ExpBetaEl>,
                  //
                  ExpSigmaPi,
                  ExpSigmaKa,
                  ExpSigmaPr,
                  //
                  NSigmaPi,
                  NSigmaKa,
                  NSigmaPr);
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
      tofpid(BETA,
             BETAERROR,
             //
             EXPBETAEL,
             EXPBETAELERROR,
             BETA > 0 ? (BETA - EXPBETAEL) / sqrt(BETAERROR * BETAERROR + EXPBETAELERROR * EXPBETAELERROR) : -999,
             //
             resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), kPionMass),
             resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), kKaonMass),
             resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes(), kProtonMass),
            //  resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes0(), kPionMass),
            //  resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes0(), kKaonMass),
            //  resp.mParam.GetExpectedSigma(i.p(), i.tofSignal(), collision.collisionTimeRes0(), kProtonMass),
             //
             0,0,0
            //  resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPi(), collision.collisionTime0(), collision.collisionTimeRes0(), kPionMass),
            //  resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpKa(), collision.collisionTime0(), collision.collisionTimeRes0(), kKaonMass),
            //  resp.mParam.GetNSigma(i.p(), i.tofSignal(), i.tofExpPr(), collision.collisionTime0(), collision.collisionTimeRes0(), kProtonMass)
             //
      );
    }
  }
};

#endif // O2_FRAMEWORK_PIDRESPONSE_H_
