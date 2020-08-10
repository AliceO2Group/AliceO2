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
#include "Analysis/SecondaryVertexHF.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"
#include "Analysis/trackUtilities.h"
#include "Analysis/RecoDecay.h"

#include <TFile.h>
#include <TH1F.h>

#include <cmath>
#include <array>
#include <cstdlib>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

/// Reconstruction of heavy-flavour 2-prong decay candidates
struct HFCandidateCreator2Prong {
  Produces<aod::HfCandProng2> hfcandprong2;
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do validation plots"};
  OutputObj<TH1F> hvtx_x_out{TH1F("hvtx_x", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_y_out{TH1F("hvtx_y", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_z_out{TH1F("hvtx_z", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hmass2{TH1F("hmass2", "2-track inv mass", 500, 0, 5.0)};
  Configurable<double> d_bz{"d_bz", 5.0, "bz field"};
  Configurable<bool> b_propdca{"b_propdca", true,
                               "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200, "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4,
                                  "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1e-3,
                                        "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9,
                                          "stop iterations is chi2/chi2old > this"};
  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massK = RecoDecay::getMassPDG(kKPlus);

  void process(aod::Collision const& collision,
               aod::HfTrackIndexProng2 const& hftrackindexprong2s,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(d_bz);
    df.setPropagateToPCA(b_propdca);
    df.setMaxR(d_maxr);
    df.setMaxDZIni(d_maxdzini);
    df.setMinParamChange(d_minparamchange);
    df.setMinRelChi2Change(d_minrelchi2change);

    double massPiK{0};
    double massKPi{0};

    // loop over 2-prong secondary vertices
    for (auto& hfpr2 : hftrackindexprong2s) {
      auto trackparvar_p1 = getTrackParCov(hfpr2.index0());
      auto trackparvar_n1 = getTrackParCov(hfpr2.index1());
      df.setUseAbsDCA(true);
      int nCand = df.process(trackparvar_p1, trackparvar_n1);
      if (nCand == 0)
        continue;
      const auto& vtx = df.getPCACandidate();
      array<float, 3> pvec0;
      array<float, 3> pvec1;
      df.getTrack(0).getPxPyPzGlo(pvec0);
      df.getTrack(1).getPxPyPzGlo(pvec1);

      auto arrMom = array{pvec0, pvec1};
      massPiK = RecoDecay::M(arrMom, array{massPi, massK});
      massKPi = RecoDecay::M(arrMom, array{massK, massPi});

      hfcandprong2(collision.posX(), collision.posY(), collision.posZ(),
                   pvec0[0], pvec0[1], pvec0[2], pvec1[0], pvec1[1], pvec1[2],
                   vtx[0], vtx[1], vtx[2], massPiK, massKPi);
      if (b_dovalplots) {
        hvtx_x_out->Fill(vtx[0]);
        hvtx_y_out->Fill(vtx[1]);
        hvtx_z_out->Fill(vtx[2]);
        hmass2->Fill(massPiK);
        hmass2->Fill(massKPi);
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFCandidateCreator2Prong>("vertexerhf-hfcandcreator2prong")};
}
