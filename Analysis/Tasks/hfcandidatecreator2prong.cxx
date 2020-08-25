// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file hfcandidatecreator2prong.cxx
/// \brief Reconstruction of heavy-flavour 2-prong decay candidates
///
/// \author Gian Michele Innocenti <gian.michele.innocenti@cern.ch>, CERN
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "DetectorsVertexing/DCAFitterN.h"
#include "Analysis/SecondaryVertexHF.h"
#include "Analysis/trackUtilities.h"
#include "ReconstructionDataFormats/DCA.h"

using namespace o2;
using namespace o2::framework;
using std::array;

/// Reconstruction of heavy-flavour 2-prong decay candidates
struct HFCandidateCreator2Prong {
  Produces<aod::HfCandBase> rowCandidateBase;
  //Produces<aod::HfCandProng2Base> rowCandidateProng2Base; // TODO split table
  Configurable<double> magneticField{"d_bz", 5.0, "magnetic field"};
  Configurable<bool> b_propdca{"b_propdca", true, "create tracks version propagated to PCA"};
  Configurable<double> d_maxr{"d_maxr", 200., "reject PCA's above this radius"};
  Configurable<double> d_maxdzini{"d_maxdzini", 4., "reject (if>0) PCA candidate if tracks DZ exceeds threshold"};
  Configurable<double> d_minparamchange{"d_minparamchange", 1.e-3, "stop iterations if largest change of any X is smaller than this"};
  Configurable<double> d_minrelchi2change{"d_minrelchi2change", 0.9, "stop iterations is chi2/chi2old > this"};
  Configurable<bool> b_dovalplots{"b_dovalplots", true, "do validation plots"};
  OutputObj<TH1F> hvtx_x_out{TH1F("hvtx_x", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_y_out{TH1F("hvtx_y", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hvtx_z_out{TH1F("hvtx_z", "2-track vtx", 100, -0.1, 0.1)};
  OutputObj<TH1F> hmass2{TH1F("hmass2", "2-track inv mass", 500, 0., 5.0)};

  double massPi = RecoDecay::getMassPDG(kPiPlus);
  double massK = RecoDecay::getMassPDG(kKPlus);
  double massPiK{0};
  double massKPi{0};

  void process(aod::Collision const& collision,
               aod::HfTrackIndexProng2 const& rowsTrackIndexProng2,
               soa::Join<aod::Tracks, aod::TracksCov, aod::TracksExtra> const& tracks)
  {
    // 2-prong vertex fitter
    o2::vertexing::DCAFitterN<2> df;
    df.setBz(magneticField);
    df.setPropagateToPCA(b_propdca);
    df.setMaxR(d_maxr);
    df.setMaxDZIni(d_maxdzini);
    df.setMinParamChange(d_minparamchange);
    df.setMinRelChi2Change(d_minrelchi2change);
    df.setUseAbsDCA(true);

    // loop over pairs of track indeces
    for (const auto& rowTrackIndexProng2 : rowsTrackIndexProng2) {
      auto trackParVarPos1 = getTrackParCov(rowTrackIndexProng2.index0());
      auto trackParVarNeg1 = getTrackParCov(rowTrackIndexProng2.index1());

      // reconstruct the 2-prong secondary vertex
      if (df.process(trackParVarPos1, trackParVarNeg1) == 0)
        continue;
      const auto& secondaryVertex = df.getPCACandidate();
      auto trackParVar0 = df.getTrack(0);
      auto trackParVar1 = df.getTrack(1);

      // get track momenta
      array<float, 3> pvec0;
      array<float, 3> pvec1;
      trackParVar0.getPxPyPzGlo(pvec0);
      trackParVar1.getPxPyPzGlo(pvec1);

      // calculate invariant masses
      auto arrayMomenta = array{pvec0, pvec1};
      massPiK = RecoDecay::M(arrayMomenta, array{massPi, massK});
      massKPi = RecoDecay::M(arrayMomenta, array{massK, massPi});

      // get track impact parameters
      // This modifies track momenta!
      auto primaryVertex = getPrimaryVertex(collision);
      o2::dataformats::DCA impactParameter0;
      o2::dataformats::DCA impactParameter1;
      trackParVar0.propagateToDCA(primaryVertex, magneticField, &impactParameter0);
      trackParVar1.propagateToDCA(primaryVertex, magneticField, &impactParameter1);

      // TODO
      float dcaDaughters = 0.;
      float errorDecayLength = 1., errorDecayLengthXY = 1.;

      // fill candidate table rows
      rowCandidateBase(collision.posX(), collision.posY(), collision.posZ(),
                       secondaryVertex[0], secondaryVertex[1], secondaryVertex[2],
                       errorDecayLength, errorDecayLengthXY, //);
                                                             //rowCandidateProng2Base( // TODO split table
                       pvec0[0], pvec0[1], pvec0[2],
                       pvec1[0], pvec1[1], pvec1[2],
                       impactParameter0.getY(), impactParameter1.getY(),
                       std::sqrt(impactParameter0.getSigmaY2()), std::sqrt(impactParameter1.getSigmaY2()),
                       dcaDaughters);

      // fill histograms
      if (b_dovalplots) {
        hvtx_x_out->Fill(secondaryVertex[0]);
        hvtx_y_out->Fill(secondaryVertex[1]);
        hvtx_z_out->Fill(secondaryVertex[2]);
        hmass2->Fill(massPiK);
        hmass2->Fill(massKPi);
      }
    }
  }
};

/// Extends the base table with expression columns.
struct HFCandidateCreator2ProngExpressions {
  Spawns<aod::HfCandProng2Ext> rowCandidateProng2;
  void init(InitContext const&) {}
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<HFCandidateCreator2Prong>("vertexerhf-hfcandcreator2prong"),
    adaptAnalysisTask<HFCandidateCreator2ProngExpressions>("vertexerhf-hfcandcreator2prong-expressions")};
}
