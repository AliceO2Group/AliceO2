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
  bool useAbsDCA = true;        ///< use abs dca minimization
  bool selectBestV0 = false;    ///< match only the best v0 for each cascade candidate
  float maxChi2 = 2.;           ///< max dca from prongs to vertex
  float minParamChange = 1e-3;  ///< stop when tracks X-params being minimized change by less that this value
  float minRelChi2Change = 0.9; ///< stop when chi2 changes by less than this value
  float maxDZIni = 5.;          ///< don't consider as a seed (circles intersection) if Z distance exceeds this
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
  float minDCAToPV = 0.1;                 ///< min DCA to PV of single track to accept
  float minRToMeanVertex = 0.5;           ///< min radial distance of V0 from beam line (mean vertex)
  float maxDCAXYToMeanVertex = 0.2;       ///< max DCA of V0 from beam line (mean vertex) for prompt V0 candidates
  float maxDCAXYToMeanVertexV0Casc = 0.5; ///< max DCA of V0 from beam line (mean vertex) for cascade V0 candidates
  float minPtV0 = 0.01;                   ///< v0 minimum pT
  float maxTglV0 = 2.;                    ///< maximum tgLambda of V0

  float causalityRTolerance = 1.; ///< V0 radius cannot exceed its contributors minR by more than this value
  float maxV0ToProngsRDiff = 50.; ///< V0 radius cannot be lower than this ammount wrt minR of contributors

  float minCosPAXYMeanVertex = 0.95;      ///< min cos of PA to beam line (mean vertex) in tr. plane for prompt V0 candidates
  float minCosPAXYMeanVertexCascV0 = 0.8; ///< min cos of PA to beam line (mean vertex) in tr. plane for V0 of cascade cand.

  float maxRToMeanVertexCascV0 = 80; // don't consider as a cascade V0 seed if above this R
  float minCosPACascV0 = 0.8;        // min cos of pointing angle to PV for cascade V0 candidates
  float minCosPA = 0.9;              ///< min cos of PA to PV for prompt V0 candidates

  float minRDiffV0Casc = 0.2; ///< cascade should be at least this radial distance below V0
  float maxRIniCasc = 90.;    // don't consider as a cascade seed (circles/line intersection) if its R exceeds this

  float maxDCAXYCasc = 0.3; // max DCA of cascade to PV in XY // TODO RS: shall we use real chi2 to vertex?
  float maxDCAZCasc = 0.3;  // max DCA of cascade to PV in Z
  float minCosPACasc = 0.7; ///< min cos of PA to PV for cascade candidates
  float minPtCasc = 0.01;   // cascade minimum pT
  float maxTglCasc = 2.;    // maximum tgLambda of cascade

  // cuts on different V0 PID params
  bool checkV0Hypothesis = true;
  float pidCutsPhoton[SVertexHypothesis::NPIDParams] = {0.001, 20, 0.60, 0.0};   // Photon
  float pidCutsK0[SVertexHypothesis::NPIDParams] = {0.003, 20, 0.07, 0.5};       // K0
  float pidCutsLambda[SVertexHypothesis::NPIDParams] = {0.001, 20, 0.07, 0.5};   // Lambda
  float pidCutsHTriton[SVertexHypothesis::NPIDParams] = {0.0025, 14, 0.07, 0.5}; // HyperTriton
  float pidCutsHhydrog4[SVertexHypothesis::NPIDParams] = {0.0025, 14, 0.07, 0.5}; // Hyperhydrog4 - Need to update
  //
  // cuts on different Cascade PID params
  bool checkCascadeHypothesis = true;
  float pidCutsXiMinus[SVertexHypothesis::NPIDParams] = {0.001, 20, 0.07, 0.5};    // XiMinus
  float pidCutsOmegaMinus[SVertexHypothesis::NPIDParams] = {0.001, 20, 0.07, 0.5}; // OmegaMinus

  O2ParamDef(SVertexerParams, "svertexer");
};

} // namespace vertexing
} // end namespace o2

#endif
