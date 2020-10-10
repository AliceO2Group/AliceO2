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
struct lambdakzeroproducer {

  Produces<aod::V0Data> v0data;

  OutputObj<TH1F> hEventCounter{TH1F("hEventCounter", "", 1, 0, 1)};
  OutputObj<TH1F> hCascCandidate{TH1F("hCascCandidate", "", 10, 0, 10)};

  //Configurables
  Configurable<double> d_bz{"d_bz", -5.0, "bz field"};
  Configurable<double> d_UseAbsDCA{"d_UseAbsDCA", kTRUE, "Use Abs DCAs"};

  double massPi = TDatabasePDG::Instance()->GetParticle(kPiPlus)->Mass();
  double massKa = TDatabasePDG::Instance()->GetParticle(kKPlus)->Mass();
  double massPr = TDatabasePDG::Instance()->GetParticle(kProton)->Mass();

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

  void process(aod::Collision const& collision, aod::V0s const& V0s, aod::FullTracks const& tracks)
  {
    //Define o2 fitter, 2-prong
    o2::vertexing::DCAFitterN<2> fitter;
    fitter.setBz(d_bz);
    fitter.setPropagateToPCA(true);
    fitter.setMaxR(200.);
    fitter.setMinParamChange(1e-3);
    fitter.setMinRelChi2Change(0.9);
    fitter.setMaxDZIni(1e9);
    fitter.setMaxChi2(1e9);
    fitter.setUseAbsDCA(d_UseAbsDCA);

    hEventCounter->Fill(0.5);
    std::array<float, 3> pVtx = {collision.posX(), collision.posY(), collision.posZ()};

    for (auto& V0 : V0s) {
      std::array<float, 3> pos = {0.};
      std::array<float, 3> pvec0 = {0.};
      std::array<float, 3> pvec1 = {0.};

      hCascCandidate->Fill(0.5);
      auto pTrack = getTrackParCov(V0.posTrack());
      auto nTrack = getTrackParCov(V0.negTrack());
      int nCand = fitter.process(pTrack, nTrack);
      if (nCand != 0) {
        fitter.propagateTracksToVertex();
        hCascCandidate->Fill(2.5);
        const auto& vtx = fitter.getPCACandidate();
        for (int i = 0; i < 3; i++)
          pos[i] = vtx[i];
        fitter.getTrack(0).getPxPyPzGlo(pvec0);
        fitter.getTrack(1).getPxPyPzGlo(pvec1);
      }

      v0data(pos[0], pos[1], pos[2],
             pvec0[0], pvec0[1], pvec0[2],
             pvec1[0], pvec1[1], pvec1[2],
             fitter.getChi2AtPCACandidate(),
             getdcaXY(V0.posTrack(), pVtx),
             getdcaXY(V0.negTrack(), pVtx));
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<lambdakzeroproducer>("lf-lambdakzeroproducer")};
}
