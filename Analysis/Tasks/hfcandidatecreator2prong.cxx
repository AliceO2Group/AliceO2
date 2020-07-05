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

    for (auto& hfpr2 : hftrackindexprong2s) {
      float x_p1 = hfpr2.index0().x();
      float alpha_p1 = hfpr2.index0().alpha();
      std::array<float, 5> arraypar_p1 = {hfpr2.index0().y(), hfpr2.index0().z(), hfpr2.index0().snp(),
                                          hfpr2.index0().tgl(), hfpr2.index0().signed1Pt()};
      std::array<float, 15> covpar_p1 = {hfpr2.index0().cYY(), hfpr2.index0().cZY(), hfpr2.index0().cZZ(),
                                         hfpr2.index0().cSnpY(), hfpr2.index0().cSnpZ(),
                                         hfpr2.index0().cSnpSnp(), hfpr2.index0().cTglY(), hfpr2.index0().cTglZ(),
                                         hfpr2.index0().cTglSnp(), hfpr2.index0().cTglTgl(),
                                         hfpr2.index0().c1PtY(), hfpr2.index0().c1PtZ(), hfpr2.index0().c1PtSnp(),
                                         hfpr2.index0().c1PtTgl(), hfpr2.index0().c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar_p1(x_p1, alpha_p1, arraypar_p1, covpar_p1);

      float x_n1 = hfpr2.index1().x();
      float alpha_n1 = hfpr2.index1().alpha();
      std::array<float, 5> arraypar_n1 = {hfpr2.index1().y(), hfpr2.index1().z(), hfpr2.index1().snp(),
                                          hfpr2.index1().tgl(), hfpr2.index1().signed1Pt()};
      std::array<float, 15> covpar_n1 = {hfpr2.index1().cYY(), hfpr2.index1().cZY(), hfpr2.index1().cZZ(),
                                         hfpr2.index1().cSnpY(), hfpr2.index1().cSnpZ(),
                                         hfpr2.index1().cSnpSnp(), hfpr2.index1().cTglY(), hfpr2.index1().cTglZ(),
                                         hfpr2.index1().cTglSnp(), hfpr2.index1().cTglTgl(),
                                         hfpr2.index1().c1PtY(), hfpr2.index1().c1PtZ(), hfpr2.index1().c1PtSnp(),
                                         hfpr2.index1().c1PtTgl(), hfpr2.index1().c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar_n1(x_n1, alpha_n1, arraypar_n1, covpar_n1);
      df.setUseAbsDCA(true);
      int nCand = df.process(trackparvar_p1, trackparvar_n1);
      if (nCand == 0)
        continue;
      const auto& vtx = df.getPCACandidate();
      std::array<float, 3> pvec0;
      std::array<float, 3> pvec1;
      df.getTrack(0).getPxPyPzGlo(pvec0);
      df.getTrack(1).getPxPyPzGlo(pvec1);
      float masspion = 0.140;
      float masskaon = 0.494;
      float mass_ = sqrt(invmass2prongs2(pvec0[0], pvec0[1],
                                         pvec0[2], masspion,
                                         pvec1[0], pvec1[1],
                                         pvec1[2], masskaon));
      float masssw_ = sqrt(invmass2prongs2(pvec0[0], pvec0[1],
                                           pvec0[2], masskaon,
                                           pvec1[0], pvec1[1],
                                           pvec1[2], masspion));
      hfcandprong2(collision.posX(), collision.posY(), collision.posZ(),
                   pvec0[0], pvec0[1], pvec0[2], pvec1[0], pvec1[1], pvec1[2],
                   vtx[0], vtx[1], vtx[2], mass_, masssw_);
      if (b_dovalplots == true) {
        hvtx_x_out->Fill(vtx[0]);
        hvtx_y_out->Fill(vtx[1]);
        hvtx_z_out->Fill(vtx[2]);
        hmass2->Fill(mass_);
        hmass2->Fill(masssw_);
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFCandidateCreator2Prong>("vertexerhf-hfcandcreator2prong")};
}
