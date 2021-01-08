// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_PVERTEXER_PARAMS_H
#define ALICEO2_PVERTEXER_PARAMS_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace vertexing
{

// These are configurable params for Primary Vertexer
struct PVertexerParams : public o2::conf::ConfigurableParamHelper<PVertexerParams> {
  static constexpr float kDefTukey = 5.0f; ///< def.value for tukey constant

  // DBSCAN clustering settings
  float dbscanMaxDist2 = 9.;   ///< distance^2 cut (eps^2).
  float dbscanDeltaT = 10.;    ///< abs. time difference cut
  float dbscanAdaptCoef = 0.1; ///< adapt dbscan minPts for each cluster as minPts=max(minPts, currentSize*dbscanAdaptCoef).

  int maxVerticesPerCluster = 1; ///< max vertices per time-z cluster to look for
  int maxTrialsPerCluster = 3;   ///< max unsucessful trials for vertex search per vertex

  // track selection
  float dcaTolerance = 1.3; ///< consider tracks within this abs DCA to mean vertex
  float pullIniCut = 9;     ///< cut on pull (n^2 sigma) on dca to mean vertex

  // parameters
  float zHistoRange = 20.;           ///< +-range of the Zseed histo
  float zHistoBinSize = 0.5;         ///< size of the Zseed histo bin
  float tukey = kDefTukey;           ///< 1./[Tukey parameter]^2
  float minScale2 = 1.;              ///< min slaling factor^2
  float acceptableScale2 = 4.;       ///< if below this factor, try to refit with minScale2
  float maxScale2 = 1.e6;            ///< max slaling factor^2
  float upscaleFactor = 9.;          ///< factor for upscaling if not candidate is found
  float slowConvergenceFactor = 0.5; ///< consider convergence as slow if ratio new/old scale2 exceeds it
  //
  // validation with FT0
  bool requireFT0ValidTimeMean = false; //true;///< require both FT0A/C
  int minNContributorsForFT0cut = 4;    ///< do not apply FT0 cut to vertice below FT0 efficiency threshold
  float maxTError = 0.2;                ///< use min of vertex time error or this for nsigma evaluation
  float minTError = 0.003;              ///< don't use error smaller than that (~BC/2/minNContributorsForFT0cut)
  float nSigmaTimeCut = 4.;             ///< eliminate vertex if there is no FT0 signal within this cut
  float timeBiasMS = 0;                 ///< relative bias in ms to add to TPCITS-based time stamp

  //
  // stopping condition params
  float maxChi2Mean = 10.;          ///< max mean chi2 of vertex to accept
  int minTracksPerVtx = 2;          ///< min N tracks per vertex
  int maxIterations = 20;           ///< max iterations per vertex fit
  int maxNScaleIncreased = 2;       ///< max number of scaling-non-decreasing iterations
  int maxNScaleSlowConvergence = 3; ///< max number of weak scaling decrease iterations
  bool useTimeInChi2 = true;        ///< use track-vertex time difference in chi2 calculation

  O2ParamDef(PVertexerParams, "pvertexer");
};

} // namespace vertexing
} // end namespace o2

#endif
