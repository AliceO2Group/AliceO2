// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "Analysis/HFSecondaryVertex.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"
#include "Analysis/RecoDecay.h"
#include "Analysis/trackUtilities.h"
#include "Analysis/StrangenessTables.h"

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <Math/Vector4D.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>
#include <cmath>
#include <array>
#include <cstdlib>
#include "PID/PIDResponse.h"
#include "Framework/ASoAHelpers.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

/// Cascade builder task: rebuilds cascades
struct cascadeproducer {
  Produces<aod::CascData> cascdata;

  OutputObj<TH1F> hEventCounter{TH1F("hEventCounter", "", 1, 0, 1)};
  OutputObj<TH1F> hCascCandidate{TH1F("hCascCandidate", "", 10, 0, 10)};

  //Configurables
  Configurable<double> d_bz{"d_bz", -5.0, "bz field"};
  Configurable<double> d_UseAbsDCA{"d_UseAbsDCA", kTRUE, "Use Abs DCAs"};

  /// Extracts dca in the XY plane
  /// \return dcaXY
  template <typename T, typename U>
  auto getdcaXY(const T& track, const U& coll)
  {
    //Calculate DCAs
    auto sinAlpha = sin(track.alpha());
    auto cosAlpha = cos(track.alpha());
    auto globalX = track.x() * cosAlpha - track.y() * sinAlpha;
    auto globalY = track.x() * sinAlpha + track.y() * cosAlpha;
    return sqrt(pow((globalX - coll[0]), 2) +
                pow((globalY - coll[1]), 2));
  }

  void process(aod::Collision const& collision, aod::V0s const& V0s, aod::Cascades const& Cascades, aod::FullTracks const& trackss)
  {
    //Define o2 fitter, 2-prong
    o2::vertexing::DCAFitterN<2> fitterV0, fitterCasc;
    fitterV0.setBz(d_bz);
    fitterV0.setPropagateToPCA(true);
    fitterV0.setMaxR(200.);
    fitterV0.setMinParamChange(1e-3);
    fitterV0.setMinRelChi2Change(0.9);
    fitterV0.setMaxDZIni(1e9);
    fitterV0.setMaxChi2(1e9);
    fitterV0.setUseAbsDCA(d_UseAbsDCA);

    fitterCasc.setBz(d_bz);
    fitterCasc.setPropagateToPCA(true);
    fitterCasc.setMaxR(200.);
    fitterCasc.setMinParamChange(1e-3);
    fitterCasc.setMinRelChi2Change(0.9);
    fitterCasc.setMaxDZIni(1e9);
    fitterCasc.setMaxChi2(1e9);
    fitterCasc.setUseAbsDCA(d_UseAbsDCA);

    hEventCounter->Fill(0.5);
    std::array<float, 3> pVtx = {collision.posX(), collision.posY(), collision.posZ()};

    for (auto& casc : Cascades) {
      auto charge = -1;
      std::array<float, 3> pos = {0.};
      std::array<float, 3> posXi = {0.};
      std::array<float, 3> pvecpos = {0.};
      std::array<float, 3> pvecneg = {0.};
      std::array<float, 3> pvecbach = {0.};

      hCascCandidate->Fill(0.5);

      //Acquire basic tracks
      auto pTrack = getTrackParCov(casc.v0().posTrack());
      auto nTrack = getTrackParCov(casc.v0().negTrack());
      auto bTrack = getTrackParCov(casc.bachelor());
      if (casc.bachelor().signed1Pt() > 0) {
        charge = +1;
      }

      int nCand = fitterV0.process(pTrack, nTrack);
      if (nCand != 0) {
        fitterV0.propagateTracksToVertex();
        hCascCandidate->Fill(1.5);
        const auto& v0vtx = fitterV0.getPCACandidate();
        for (int i = 0; i < 3; i++) {
          pos[i] = v0vtx[i];
        }

        std::array<float, 21> cov0 = {0};
        std::array<float, 21> cov1 = {0};
        std::array<float, 21> covV0 = {0};

        //Covariance matrix calculation
        const int momInd[6] = {9, 13, 14, 18, 19, 20}; // cov matrix elements for momentum component
        fitterV0.getTrack(0).getPxPyPzGlo(pvecpos);
        fitterV0.getTrack(1).getPxPyPzGlo(pvecneg);
        fitterV0.getTrack(0).getCovXYZPxPyPzGlo(cov0);
        fitterV0.getTrack(1).getCovXYZPxPyPzGlo(cov1);
        for (int i = 0; i < 6; i++) {
          int j = momInd[i];
          covV0[j] = cov0[j] + cov1[j];
        }
        auto covVtxV0 = fitterV0.calcPCACovMatrix();
        covV0[0] = covVtxV0(0, 0);
        covV0[1] = covVtxV0(1, 0);
        covV0[2] = covVtxV0(1, 1);
        covV0[3] = covVtxV0(2, 0);
        covV0[4] = covVtxV0(2, 1);
        covV0[5] = covVtxV0(2, 2);

        const std::array<float, 3> vertex = {(float)v0vtx[0], (float)v0vtx[1], (float)v0vtx[2]};
        const std::array<float, 3> momentum = {pvecpos[0] + pvecneg[0], pvecpos[1] + pvecneg[1], pvecpos[2] + pvecneg[2]};

        auto tV0 = o2::track::TrackParCov(vertex, momentum, covV0, 0);
        tV0.setQ2Pt(0); //No bending, please
        int nCand2 = fitterCasc.process(tV0, bTrack);
        if (nCand2 != 0) {
          fitterCasc.propagateTracksToVertex();
          hCascCandidate->Fill(2.5);
          const auto& cascvtx = fitterCasc.getPCACandidate();
          for (int i = 0; i < 3; i++) {
            posXi[i] = cascvtx[i];
          }
          fitterCasc.getTrack(1).getPxPyPzGlo(pvecbach);
        } //end if cascade recoed
      }   //end if v0 recoed
      //Fill table, please
      cascdata(charge, posXi[0], posXi[1], posXi[2], pos[0], pos[1], pos[2],
               pvecpos[0], pvecpos[1], pvecpos[2],
               pvecneg[0], pvecneg[1], pvecneg[2],
               pvecbach[0], pvecbach[1], pvecbach[2],
               fitterV0.getChi2AtPCACandidate(), fitterCasc.getChi2AtPCACandidate(),
               getdcaXY(casc.v0().posTrack(), pVtx),
               getdcaXY(casc.v0().negTrack(), pVtx),
               getdcaXY(casc.bachelor(), pVtx));
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<cascadeproducer>("lf-cascadeproducer")};
}
