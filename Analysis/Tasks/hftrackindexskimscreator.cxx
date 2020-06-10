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
#include "Analysis/SecondaryVertex.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "ReconstructionDataFormats/Track.h"

#include <TFile.h>
#include <TH1F.h>
#include <cmath>
#include <array>
#include <cstdlib>

using namespace o2;
using namespace o2::framework;

struct HFTrackIndexSkimsCreator {

  Produces<aod::SecVtx2Prong> secvtx2prong;
  Produces<aod::SecVtx3Prong> secvtx3prong;
  Configurable<int> triggerindex{"triggerindex", -1, "trigger index"};
  Configurable<double> ptmintrack{"ptmintrack", -1, "ptmin single track"};
  Configurable<int> do3prong{"do3prong", 0, "do 3 prong"};
  Configurable<double> d_bz{"d_bz", 5.0, "bz field"};
  Configurable<bool> b_propdca{"b_propdca", true, \
                "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200, "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4, \
                "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1e-3, \
                "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change {"d_minrelchi2change", 0.9, \
                "stop iterations is chi2/chi2old > this"};
  Configurable<int> d_tpcnclsfound {"d_tpcnclsfound", 70, "min number of tpc cls >="};

  //Configurable<double> {"", , ""};
  //Configurable<int> {"", , ""};
  //Configurable<bool> {"", , ""};

  void process(aod::Collision const& collision,
               aod::BCs const& bcs,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
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
      if (abs(track_0.signed1Pt())<ptmintrack)
        continue;

      UChar_t clustermap_0 = track_0.itsClusterMap();
      //fill track distribution before selection
      bool isselected_0 = track_0.tpcNClsFound() >= d_tpcnclsfound && track_0.flags() & 0x4;
      isselected_0 = isselected_0 && (TESTBIT(clustermap_0, 0) || TESTBIT(clustermap_0, 1));
      if (!isselected_0)
        continue;

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
        if (abs(track_1.signed1Pt())<ptmintrack)
          continue;
        UChar_t clustermap_1 = track_1.itsClusterMap();
        bool isselected_1 = track_1.tpcNClsFound() >= d_tpcnclsfound && track_1.flags() & 0x4;
        isselected_1 = isselected_1 && (TESTBIT(clustermap_1, 0) || TESTBIT(clustermap_1, 1));
        if (!isselected_1)
          continue;
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
        o2::track::TrackParCov trackdec0 = df.getTrack(0);
        o2::track::TrackParCov trackdec1 = df.getTrack(1);
        std::array<float, 3> pvec0;
        std::array<float, 3> pvec1;
        trackdec0.getPxPyPzGlo(pvec0);
        trackdec1.getPxPyPzGlo(pvec1);
        float masspion = 0.140;
        float masskaon = 0.494;
        float mass_ = invmass2prongs(pvec0[0], pvec0[1], pvec0[2], masspion,
                                     pvec1[0], pvec1[1], pvec1[2], masskaon);
        float masssw_ = invmass2prongs(pvec0[0], pvec0[1], pvec0[2], masskaon,
                                       pvec1[0], pvec1[1], pvec1[2], masspion);
        secvtx2prong(track_0.collisionId(),
                     collision.posX(), collision.posY(), collision.posZ(),
                     vtx[0], vtx[1], vtx[2], track_0.globalIndex(),
                     pvec0[0], pvec0[1], pvec0[2], track_0.y(),
                     track_1.globalIndex(), pvec1[0], pvec1[1], pvec1[2], track_1.y(),
                     mass_, masssw_);
        if (do3prong == 1) {
          for (auto it2 = it0 + 1; it2 != tracks.end(); ++it2) {
            if(it2 == it0 || it2 == it1)
              continue;
            auto& track_2 = *it2;
            if (abs(track_2.signed1Pt())<ptmintrack)
              continue;
            UChar_t clustermap_2 = track_2.itsClusterMap();
            bool isselected_2 = track_2.tpcNClsFound() >= d_tpcnclsfound && track_2.flags() & 0x4;
            isselected_2 = isselected_2 && (TESTBIT(clustermap_2, 0) || TESTBIT(clustermap_2, 1));
            if (!isselected_2)
              continue;
            if (track_1.signed1Pt() * track_2.signed1Pt() > 0)
              continue;
            float x2_ = track_2.x();
            float alpha2_ = track_2.alpha();
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
            o2::track::TrackParCov trackdec0_3p = df3.getTrack(0);
            o2::track::TrackParCov trackdec1_3p = df3.getTrack(1);
            o2::track::TrackParCov trackdec2_3p = df3.getTrack(2);
            std::array<float, 3> pvec0_3p;
            std::array<float, 3> pvec1_3p;
            std::array<float, 3> pvec2_3p;
            trackdec0_3p.getPxPyPzGlo(pvec0_3p);
            trackdec1_3p.getPxPyPzGlo(pvec1_3p);
            trackdec2_3p.getPxPyPzGlo(pvec2_3p);
            secvtx3prong(track_0.collisionId(),
                         collision.posX(), collision.posY(), collision.posZ(),
                         vtx3[0], vtx3[1], vtx3[2],
                         track_0.globalIndex(), pvec0_3p[0], pvec0_3p[1], pvec0_3p[2], track_0.y(),
                         track_1.globalIndex(), pvec1_3p[0], pvec1_3p[1], pvec1_3p[2], track_1.y(),
                         track_2.globalIndex(), pvec2_3p[0], pvec2_3p[1], pvec2_3p[2], track_2.y(),
                         -1., -1.);
          }
        }
      }
    }
  }
};


WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFTrackIndexSkimsCreator>("vertexerhf-hftrackindexskimscreator")};
}
