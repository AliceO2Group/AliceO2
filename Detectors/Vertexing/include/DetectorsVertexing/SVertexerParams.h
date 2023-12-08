// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SVertexerParams.h
/// \brief Configurable params for secondary vertexer
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_SVERTEXER_PARAMS_H
#define ALICEO2_SVERTEXER_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "DetectorsVertexing/SVertexHypothesis.h"
#include "DetectorsBase/Propagator.h"

namespace o2
{
namespace vertexing
{

// These are configurable params for Primary Vertexer
struct SVertexerParams : public o2::conf::ConfigurableParamHelper<SVertexerParams> {

  // parameters
  bool createFullV0s = false;      ///< fill V0s prongs/kinematics
  bool createFullCascades = false; ///< fill cascades prongs/kinematics
  bool createFull3Bodies = false;  ///< fill 3-body decays prongs/kinematics

  bool useAbsDCA = true;        ///< use abs dca minimization
  bool selectBestV0 = false;    ///< match only the best v0 for each cascade candidate
  float maxChi2 = 2.;           ///< max dca from prongs to vertex
  float minParamChange = 1e-3;  ///< stop when tracks X-params being minimized change by less that this value
  float minRelChi2Change = 0.9; ///< stop when chi2 changes by less than this value
  float maxDZIni = 5.;          ///< don't consider as a seed (circles intersection) if Z distance exceeds this
  float maxDXYIni = 4.;         ///< don't consider as a seed (circles intersection) if XY distance exceeds this
  float maxRIni = 150;          ///< don't consider as a seed (circles intersection) if its R exceeds this
  //
  // propagation options
  int matCorr = int(o2::base::Propagator::MatCorrType::USEMatCorrNONE); ///< material correction to use
  float minRFor3DField = 40;                                            ///< above this radius use 3D field
  float maxStep = 2.;                                                   ///< max step size when external propagator is used
  float maxSnp = 0.95;                                                  ///< max snp when external propagator is used
  float minXSeed = -1.;                                                 ///< minimal X of seed in prong frame (within the radial resolution track should not go to negative X)
  bool usePropagator = false;                                           ///< use external propagator
  bool refitWithMatCorr = false;                                        ///< refit V0 applying material corrections
  //
  int maxPVContributors = 2;              ///< max number PV contributors to allow in V0
  float minDCAToPV = 0.05;                ///< min DCA to PV of single track to accept
  float minRToMeanVertex = 0.5;           ///< min radial distance of V0 from beam line (mean vertex)
  float maxDCAXYToMeanVertex = 0.2;       ///< max DCA of V0 from beam line (mean vertex) for prompt V0 candidates
  float maxDCAXYToMeanVertexV0Casc = 2.;  ///< max DCA of V0 from beam line (mean vertex) for cascade V0 candidates
  float maxDCAXYToMeanVertex3bodyV0 = 2.; ///< max DCA of V0 from beam line (mean vertex) for 3body V0 candidates
  float minPtV0 = 0.01;                   ///< v0 minimum pT
  float minPtV0FromCascade = 0.3;         ///< v0 minimum pT for v0 to be used in cascading (lowest pT Run 2 lambda: 0.4)
  float maxTglV0 = 2.;                    ///< maximum tgLambda of V0

  float causalityRTolerance = 1.; ///< V0 radius cannot exceed its contributors minR by more than this value
  float maxV0ToProngsRDiff = 50.; ///< V0 radius cannot be lower than this ammount wrt minR of contributors

  float minCosPAXYMeanVertex = 0.95;       ///< min cos of PA to beam line (mean vertex) in tr. plane for prompt V0 candidates
  float minCosPAXYMeanVertexCascV0 = 0.8;  ///< min cos of PA to beam line (mean vertex) in tr. plane for V0 of cascade cand.
  float minCosPAXYMeanVertex3bodyV0 = 0.9; ///< min cos of PA to beam line (mean vertex) in tr. plane for 3body V0 cand.

  float maxRToMeanVertexCascV0 = 80; // don't consider as a cascade V0 seed if above this R
  float minCosPACascV0 = 0.8;        // min cos of pointing angle to PV for cascade V0 candidates
  float minCosPA3bodyV0 = 0.8;       // min cos of PA to PV for 3body V0
  float minCosPA = 0.9;              ///< min cos of PA to PV for prompt V0 candidates

  float minRDiffV0Casc = 0.2; ///< cascade should be at least this radial distance below V0
  float maxRDiffV03body = 3;  ///< Maximum difference between V0 and 3body radii
  float maxRIniCasc = 90.;    // don't consider as a cascade seed (circles/line intersection) if its R exceeds this

  float maxDCAXYCasc = 0.3;  // max DCA of cascade to PV in XY // TODO RS: shall we use real chi2 to vertex?
  float maxDCAZCasc = 0.3;   // max DCA of cascade to PV in Z
  float maxDCAXY3Body = 0.5; // max DCA of 3 body decay to PV in XY // TODO RS: shall we use real chi2 to vertex?
  float maxDCAZ3Body = 1.;   // max DCA of 3 body decay to PV in Z
  float minCosPACasc = 0.7;  // min cos of PA to PV for cascade candidates
  float minCosPA3body = 0.8; // min cos of PA to PV for 3body decay candidates
  float minPtCasc = 0.01;    // cascade minimum pT
  float maxTglCasc = 2.;     // maximum tgLambda of cascade
  float minPt3Body = 0.01;   // minimum pT of 3body V0
  float maxTgl3Body = 2.;    // maximum tgLambda of 3body V0

  float maxRIni3body = 90.; // don't consider as a 3body seed (circles/line intersection) if its R exceeds this

  // Cuts for TPC only tracks
  bool mExcludeTPCtracks = false;            // don't loop over TPC tracks if true (if loaded, dEdx info is used instead)
  float mTPCTrackMaxX = -1.;                 // don't use TPC standalone tracks with X exceeding this;
  float mTPCTrack2Beam = 21.f;               // straight line for TPC track back to beamline
  bool mTPCTrackPhotonTune = true;           // use TPC-only photon tuning
  int mTPCTrackMinNClusters = -1;            // minimum number of clusters
  float mTPCTrackXY2Radius = 90.f;           // check for realistic conversion point, e.g., in material
  float mTPCTrackD2R = 4.f;                  // check for maximal distance of pairs and their radii
  float mTPCTrackDR = 90.f;                  // check if preliminary conversion point can lie somewhere reasonable
  float minTPCdEdx = 250;                    // starting from this dEdx value, tracks with p > minMomTPCdEdx are always accepted
  float minMomTPCdEdx = 0.8;                 // minimum p for tracks with dEdx > mMinTPCdEdx to be accepted
  float maxV0TglAbsDiff = 0.3;               ///< max absolute difference in Tgl for V0 for photons only
  float mTPCTrackMaxChi2 = 4.;               ///< max DCA from prongs to vertex for photon TPC-only track only
  float mTPCTrackMaxDZIni = 8.;              ///< don't consider as a seed (circles intersection) if Z distance exceeds this, for photon TPC-only track only
  float mTPCTrackMaxDXYIni = 8.;             ///< don't consider as a seed (circles intersection) if XY distance exceeds this, for photon TPC-only track only
  float mTPCTrackMaxDCAXY2ToMeanVertex = 2.; ///< max DCA^2 of V0 from beam line (mean vertex) for prompt V0 candidates, for photon TPC-only track only

  bool mITSTrackPhotonTune = true;       // use ITS-only photon tuning
  int8_t mITSOBminNclu = 4;              // allow with ITS photon tune, ITS tracks with at least this many clusters
  int8_t mITSSAminNclu = 6;              // global requirement of at least this many ITS clusters if no TPC info present (N.B.: affects all secondary vertexing)
  int8_t mITSSAminNcluCascades = 6;      // require at least this many ITS clusters if no TPC info present for cascade finding.
  bool mRequireTPCforCascBaryons = true; // require that baryon daughter of cascade has TPC
  bool mSkipTPCOnlyCascade = true;       // skip TPC only tracks when doing cascade finding
  bool mSkipTPCOnly3Body = true;         // skip TPC only tracks when doing cascade finding

  // percent deviation from expected proton dEdx - to be replaced - estimated sigma from TPC for now 6%; a 6*sigma cut is therefore 36% = 0.36f. Any value above 1.0f will be ignored manually when checking.
  float mFractiondEdxforCascBaryons = 0.36f;
  // default: average 2023 from C. Sonnabend, Nov 2023: ([0.217553   4.02762    0.00850178 2.33324    0.880904  ])
  // to-do: grab from CCDB when available -> discussion with TPC experts, not available yet
  float mBBpars[5] = {0.217553, 4.02762, 0.00850178, 2.33324, 0.880904};

  // cuts on different V0 PID params
  bool checkV0Hypothesis = true;                                                                                               // enable Mass Hypothesis check
  bool checkV0AP = false;                                                                                                      // enable Armenteros-Podolanski check
  float pidCutsPhoton[SVertexHypothesis::NPIDParams] = {0.001, 20, 0.10, 20, 0.10, 0.0, 0.0, 0.0, 0.0};                        // Photon
  float pidCutsPhotonAP_qT = 0.04;                                                                                             // Loose Armenteros-Podolanski q_T cut for photons in case mass-hypothesis was good
  float pidCutsK0[SVertexHypothesis::NPIDParams] = {0., 20, 0., 5.0, 0.0, 2.84798e-03, 9.84206e-04, 3.31951e-03, 2.39438};     // K0
  float pidCutsK0AP_qT = 0.6;                                                                                                  // Loose Armenteros-Podolanski q_T cut for K0s in case mass-hypothesis was good
  float pidCutsLambda[SVertexHypothesis::NPIDParams] = {0., 20, 0., 5.0, 0.0, 1.09004e-03, 2.62291e-04, 8.93179e-03, 2.83121}; // Lambda
  float pidCutsLambdaAP_qT = 0.9;                                                                                              // Loose Armenteros-Podolanski q_T cut for Lambda in case mass-hypothesis was good
  float pidCutsLambdaAP_a = 0.9;                                                                                               // Loose Armenteros-Podolanski alpha cut for Lambda in case mass-hypothesis was good
  float pidCutsHTriton[SVertexHypothesis::NPIDParams] = {0.0025, 14, 0.07, 14, 0.0, 0.5, 0.0, 0.0, 0.0};                       // HyperTriton
  float pidCutsHhydrog4[SVertexHypothesis::NPIDParams] = {0.0025, 14, 0.07, 14, 0.0, 0.5, 0.0, 0.0, 0.0};                      // Hyperhydrog4 - Need to update
  //
  // cuts on different Cascade PID params
  bool checkCascadeHypothesis = true;
  float pidCutsXiMinus[SVertexHypothesis::NPIDParams] = {0.0, 10, 0.0, 4.0, 0.0, 1.56315e-03, 2.23279e-04, 2.75136e-02, 3.309};          // XiMinus
  float pidCutsOmegaMinus[SVertexHypothesis::NPIDParams] = {0.0, 10, 0.0, 4.0, 0.0, 1.43572e-03, 6.94416e-04, 2.13534e+05, 1.48889e+01}; // OmegaMinus
  float maximalCascadeWidth = 0.006;
  //
  // cuts on different 3 body PID params
  bool check3bodyHypothesis = true;
  float pidCutsH3L3body[SVertex3Hypothesis::NPIDParams] = {0.0025, 14, 0.07, 0.5};  // H3L -> d p pi-
  float pidCutsHe4L3body[SVertex3Hypothesis::NPIDParams] = {0.0025, 14, 0.07, 0.5}; // He4L -> He3 p pi-

  O2ParamDef(SVertexerParams, "svertexer");
};
} // namespace vertexing

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::vertexing::SVertexerParams> : std::true_type {
};
} // namespace framework

} // end namespace o2

#endif
