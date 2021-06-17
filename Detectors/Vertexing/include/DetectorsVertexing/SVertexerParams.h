// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

namespace o2
{
namespace vertexing
{

// These are configurable params for Primary Vertexer
struct SVertexerParams : public o2::conf::ConfigurableParamHelper<SVertexerParams> {

  // parameters
  float maxChi2 = 2.;           ///< max dca from prongs to vertex
  float minParamChange = 1e-3;  ///< stop when tracks X-params being minimized change by less that this value
  float minRelChi2Change = 0.9; ///< stop when chi2 changes by less than this value
  float maxDZIni = 5.;          ///< don't consider as a seed (circles intersection) if Z distance exceeds this
  float maxRIni = 150;          ///< don't consider as a seed (circles intersection) if its R exceeds this
  bool useAbsDCA = true; ///< use abs dca minimization
  //
  float minRToMeanVertex = 0.5;           ///< min radial distance of V0 from beam line (mean vertex)
  float maxDCAXYToMeanVertex = 0.2;       ///< max DCA of V0 from beam line (mean vertex) for prompt V0 candidates
  float maxDCAXYToMeanVertexV0Casc = 0.5; ///< max DCA of V0 from beam line (mean vertex) for cascade V0 candidates

  float minCosPAXYMeanVertex = 0.85;      ///< min cos of PA to beam line (mean vertex) in tr. plane for prompt V0 candidates
  float minCosPAXYMeanVertexCascV0 = 0.8; ///< min cos of PA to beam line (mean vertex) in tr. plane for V0 of cascade cand.

  float maxRToMeanVertexCascV0 = 80; // don't consider as a cascade V0 seed if above this R
  float minCosPACascV0 = 0.8;        // min cos of pointing angle to PV for cascade V0 candidates

  float minCosPA = 0.9; ///< min cos of PA to PV for prompt V0 candidates

  float minRDiffV0Casc = 0.2; ///< cascade should be at least this radial distance below V0
  float maxRIniCasc = 50.;    // don't consider as a cascade seed (circles/line intersection) if its R exceeds this

  float maxDCAXYCasc = 0.3; // max DCA of cascade to PV in XY // TODO RS: shall we use real chi2 to vertex?
  float maxDCAZCasc = 0.3;  // max DCA of cascade to PV in Z
  float minCosPACasc = 0.7; ///< min cos of PA to PV for cascade candidates

  // cuts on different V0 PID params
  float pidCutsPhoton[SVertexHypothesis::NPIDParams] = {0.001, 20, 0.60, 0.0};   // Photon
  float pidCutsK0[SVertexHypothesis::NPIDParams] = {0.003, 20, 0.07, 0.5};       // K0
  float pidCutsLambda[SVertexHypothesis::NPIDParams] = {0.001, 20, 0.07, 0.5};   // Lambda
  float pidCutsHTriton[SVertexHypothesis::NPIDParams] = {0.0025, 14, 0.07, 0.5}; // HyperTriton
  //
  // cuts on different Cascade PID params
  float pidCutsXiMinus[SVertexHypothesis::NPIDParams] = {0.001, 20, 0.07, 0.5};    // XiMinus
  float pidCutsOmegaMinus[SVertexHypothesis::NPIDParams] = {0.001, 20, 0.07, 0.5}; // OmegaMinus

  O2ParamDef(SVertexerParams, "svertexer");
};

} // namespace vertexing
} // end namespace o2

#endif
