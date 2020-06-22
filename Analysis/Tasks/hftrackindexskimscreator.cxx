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

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>
#include <cstdlib>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

namespace o2::aod
{
namespace seltrack
{
DECLARE_SOA_COLUMN(IsSel, issel, int);
DECLARE_SOA_COLUMN(DCAPrim0, dcaprim0, float);
DECLARE_SOA_COLUMN(DCAPrim1, dcaprim1, float);
} // namespace seltrack
DECLARE_SOA_TABLE(SelTrack, "AOD", "SELTRACK", seltrack::IsSel, seltrack::DCAPrim0,
                  seltrack::DCAPrim1);
} // namespace o2::aod

struct SelectTracks {
  Produces<aod::SelTrack> seltrack;
  Configurable<double> ptmintrack{"ptmintrack", -1, "ptmin single track"};
  Configurable<double> dcatrackmin{"dcatrackmin", 0, "dca single track min"};
  Configurable<int> d_tpcnclsfound{"d_tpcnclsfound", 70, "min number of tpc cls >="};
  Configurable<double> d_bz{"d_bz", 5.0, "bz field"};
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do single track val plots"};
  //OutputObj<TH1F> hdca{TH1F("hdca", "dca single tracks (cm)", 1000, 0., 1.)};

  void process(aod::Collision const& collision,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    Point3D<float> vtxXYZ(collision.posX(), collision.posY(), collision.posZ());
    for (auto it0 = tracks.begin(); it0 != tracks.end(); ++it0) {
      auto& track_0 = *it0;
      int status = 1;
      if (abs(track_0.signed1Pt()) < ptmintrack)
        status = 0;
      UChar_t clustermap_0 = track_0.itsClusterMap();
      bool isselected_0 = track_0.tpcNClsFound() >= d_tpcnclsfound && track_0.flags() & 0x4;
      isselected_0 = isselected_0 && (TESTBIT(clustermap_0, 0) || TESTBIT(clustermap_0, 1));
      if (!isselected_0)
        status = 0;
      array<float, 2> dca;
      float x0_ = track_0.x();
      float alpha0_ = track_0.alpha();
      std::array<float, 5> arraypar0 = {track_0.y(), track_0.z(), track_0.snp(),
                                        track_0.tgl(), track_0.signed1Pt()};
      std::array<float, 15> covpar0 = {track_0.cYY(), track_0.cZY(), track_0.cZZ(),
                                       track_0.cSnpY(), track_0.cSnpZ(),
                                       track_0.cSnpSnp(), track_0.cTglY(), track_0.cTglZ(),
                                       track_0.cTglSnp(), track_0.cTglTgl(),
                                       track_0.c1PtY(), track_0.c1PtZ(), track_0.c1PtSnp(),
                                       track_0.c1PtTgl(), track_0.c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar0(x0_, alpha0_, arraypar0, covpar0);
      trackparvar0.propagateParamToDCA(vtxXYZ, d_bz, &dca);
      if (dca[0] * dca[0] + dca[1] * dca[1] < dcatrackmin * dcatrackmin)
        status = 0;
      //hdca->Fill(sqrt(dca[0]*dca[0] + dca[1]*dca[1]));
      seltrack(status, dca[0], dca[1]);
    }
  }
};

struct HFTrackIndexSkimsCreator {
  float masspion = 0.140;
  float masskaon = 0.494;

  Produces<aod::HfTrackIndexProng2> hftrackindexprong2;
  Produces<aod::HfTrackIndexProng3> hftrackindexprong3;
  Configurable<int> triggerindex{"triggerindex", -1, "trigger index"};
  Configurable<int> do3prong{"do3prong", 0, "do 3 prong"};
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

  Filter seltrack = (aod::seltrack::issel == 1);

  void process(aod::Collision const& collision,
               aod::BCs const& bcs,
               soa::Filtered<soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra, aod::SelTrack>> const& tracks)
  {
    int trigindex = int{triggerindex};
    if (trigindex != -1) {
      //LOGF(info, "Selecting on trigger bit %d", trigindex);
      uint64_t triggerMask = collision.bc().triggerMask();
      bool isTriggerClassFired = triggerMask & 1ul << (trigindex - 1);
      if (!isTriggerClassFired)
        return;
    }

    LOGF(info, "N. of Tracks for collision: %d", tracks.size());
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(d_bz);
    df.setPropagateToPCA(b_propdca);
    df.setMaxR(d_maxr);
    df.setMaxDZIni(d_maxdzini);
    df.setMinParamChange(d_minparamchange);
    df.setMinRelChi2Change(d_minrelchi2change);

    o2::vertexing::DCAFitterN<3> df3;
    df3.setBz(d_bz);
    df3.setPropagateToPCA(b_propdca);
    df3.setMaxR(d_maxr);
    df3.setMaxDZIni(d_maxdzini);
    df3.setMinParamChange(d_minparamchange);
    df3.setMinRelChi2Change(d_minrelchi2change);

    for (auto it0 = tracks.begin(); it0 != tracks.end(); ++it0) {
      auto& track_0 = *it0;
      float x0_ = track_0.x();
      float alpha0_ = track_0.alpha();
      std::array<float, 5> arraypar0 = {track_0.y(), track_0.z(), track_0.snp(),
                                        track_0.tgl(), track_0.signed1Pt()};
      std::array<float, 15> covpar0 = {track_0.cYY(), track_0.cZY(), track_0.cZZ(),
                                       track_0.cSnpY(), track_0.cSnpZ(),
                                       track_0.cSnpSnp(), track_0.cTglY(), track_0.cTglZ(),
                                       track_0.cTglSnp(), track_0.cTglTgl(),
                                       track_0.c1PtY(), track_0.c1PtZ(), track_0.c1PtSnp(),
                                       track_0.c1PtTgl(), track_0.c1Pt21Pt2()};
      o2::track::TrackParCov trackparvar0(x0_, alpha0_, arraypar0, covpar0);
      for (auto it1 = it0 + 1; it1 != tracks.end(); ++it1) {
        auto& track_1 = *it1;
        if (track_0.signed1Pt() * track_1.signed1Pt() > 0)
          continue;
        float x1_ = track_1.x();
        float alpha1_ = track_1.alpha();
        std::array<float, 5> arraypar1 = {track_1.y(), track_1.z(), track_1.snp(),
                                          track_1.tgl(), track_1.signed1Pt()};
        std::array<float, 15> covpar1 = {track_1.cYY(), track_1.cZY(), track_1.cZZ(),
                                         track_1.cSnpY(), track_1.cSnpZ(),
                                         track_1.cSnpSnp(), track_1.cTglY(), track_1.cTglZ(),
                                         track_1.cTglSnp(), track_1.cTglTgl(),
                                         track_1.c1PtY(), track_1.c1PtZ(), track_1.c1PtSnp(),
                                         track_1.c1PtTgl(), track_1.c1Pt21Pt2()};
        o2::track::TrackParCov trackparvar1(x1_, alpha1_, arraypar1, covpar1);
        df.setUseAbsDCA(true);
        int nCand = df.process(trackparvar0, trackparvar1);
        if (nCand == 0)
          continue;
        const auto& vtx = df.getPCACandidate();
        //LOGF(info, "vertex x %f", vtx[0]);
        std::array<float, 3> pvec0;
        std::array<float, 3> pvec1;
        df.getTrack(0).getPxPyPzGlo(pvec0);
        df.getTrack(1).getPxPyPzGlo(pvec1);
        float mass_ = invmass2prongs(pvec0[0], pvec0[1], pvec0[2], masspion,
                                     pvec1[0], pvec1[1], pvec1[2], masskaon);
        float masssw_ = invmass2prongs(pvec0[0], pvec0[1], pvec0[2], masskaon,
                                       pvec1[0], pvec1[1], pvec1[2], masspion);
        hftrackindexprong2(track_0.collisionId(),
                           track_0.globalIndex(),
                           track_1.globalIndex(), 1.);
        if (do3prong == 1) {
          for (auto it2 = it0 + 1; it2 != tracks.end(); ++it2) {
            if (it2 == it0 || it2 == it1)
              continue;
            auto& track_2 = *it2;
            if (track_1.signed1Pt() * track_2.signed1Pt() > 0)
              continue;
            float x2_ = track_2.x();
            float alpha2_ = track_2.alpha();
            double mass3prong2 = invmass3prongs2(track_0.px(), track_0.py(), track_0.pz(), masspion,
                                                 track_1.px(), track_1.py(), track_1.pz(), masspion,
                                                 track_2.px(), track_2.py(), track_2.pz(), masspion);
            if (mass3prong2 < 1.5 * 1.5 || mass3prong2 > 2.5 * 2.5)
              continue;

            std::array<float, 5> arraypar2 = {track_2.y(), track_2.z(), track_2.snp(),
                                              track_2.tgl(), track_2.signed1Pt()};
            std::array<float, 15> covpar2 = {track_2.cYY(), track_2.cZY(), track_2.cZZ(),
                                             track_2.cSnpY(), track_2.cSnpZ(),
                                             track_2.cSnpSnp(), track_2.cTglY(), track_2.cTglZ(),
                                             track_2.cTglSnp(), track_2.cTglTgl(),
                                             track_2.c1PtY(), track_2.c1PtZ(), track_2.c1PtSnp(),
                                             track_2.c1PtTgl(), track_2.c1Pt21Pt2()};
            o2::track::TrackParCov trackparvar2(x2_, alpha2_, arraypar2, covpar2);
            df3.setUseAbsDCA(true);
            int nCand3 = df3.process(trackparvar0, trackparvar1, trackparvar2);
            if (nCand3 == 0)
              continue;
            const auto& vtx3 = df3.getPCACandidate();
            std::array<float, 3> pvec0_3p;
            std::array<float, 3> pvec1_3p;
            std::array<float, 3> pvec2_3p;
            df3.getTrack(0).getPxPyPzGlo(pvec0_3p);
            df3.getTrack(1).getPxPyPzGlo(pvec1_3p);
            df3.getTrack(2).getPxPyPzGlo(pvec2_3p);
            hftrackindexprong3(track_0.collisionId(),
                               track_0.globalIndex(),
                               track_1.globalIndex(),
                               track_2.globalIndex(), 1.);
          }
        }
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<SelectTracks>("produce-sel-track"),
    adaptAnalysisTask<HFTrackIndexSkimsCreator>("vertexerhf-hftrackindexskimscreator")};
}
