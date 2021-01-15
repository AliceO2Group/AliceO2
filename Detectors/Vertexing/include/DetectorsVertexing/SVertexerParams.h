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

namespace o2
{
namespace vertexing
{

// These are configurable params for Primary Vertexer
struct SVertexerParams : public o2::conf::ConfigurableParamHelper<SVertexerParams> {

  enum PIDV0 { Photon,
               K0,
               Lambda,
               AntiLambda,
               HyperTriton,
               AntiHyperTriton,
               NPIDV0 };
  enum PIDParams { SigmaMV0,
                   NSigmaMV0,
                   Margin,
                   CPt,
                   NPIDParams };
  // parameters
  float maxChi2 = 2.;           ///< max dca from prongs to vertex
  float minParamChange = 1e-3;  ///< stop when tracks X-params being minimized change by less that this value
  float minRelChi2Change = 0.9; ///< stop when chi2 changes by less than this value
  float maxDZIni = 5.;          ///< don't consider as a seed (circles intersection) if Z distance exceeds this
  float maxRIni = 150;          ///< don't consider as a seed (circles intersection) if its R exceeds this

  bool useAbsDCA = true; ///< use abs dca minimization
  //
  float minRfromMeanVertex = 0.5;     ///< min radial distance of V0 from beam line (mean vertex)
  float maxDCAXYfromMeanVertex = 0.2; ///< min DCA of V0 from beam line (mean vertex)
  float minCosPointingAngle = 0.8;

  // cuts on different PID params
  float pidCutsPhoton[NPIDParams] = {0.001, 20, 0.60, 0.0};   // Photon
  float pidCutsK0[NPIDParams] = {0.003, 20, 0.07, 0.5};       // K0
  float pidCutsLambda[NPIDParams] = {0.001, 20, 0.07, 0.5};   // Lambda
  float pidCutsHTriton[NPIDParams] = {0.0025, 14, 0.07, 0.5}; // HyperTriton

  O2ParamDef(SVertexerParams, "svertexer");
};

} // namespace vertexing
} // end namespace o2

#endif
