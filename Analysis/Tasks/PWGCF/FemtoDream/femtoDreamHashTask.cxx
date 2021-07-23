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

/// \file femtoDreamReaderTask.cxx
/// \brief Tasks that reads the track tables used for the pairing
/// This task is common for all femto analyses
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "include/FemtoDream/FemtoDerived.h"
#include "AnalysisCore/EventMixing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "Framework/ASoAHelpers.h"

using namespace o2;
using namespace o2::framework;

struct femtoDreamPairHashTask {

  Configurable<std::vector<float>> CfgVtxBins{"CfgVtxBins", std::vector<float>{-10.0f, -8.f, -6.f, -4.f, -2.f, 0.f, 2.f, 4.f, 6.f, 8.f, 10.f}, "Mixing bins - z-vertex"};
  Configurable<std::vector<float>> CfgMultBins{"CfgMultBins", std::vector<float>{0.0f, 20.0f, 40.0f, 60.0f, 80.0f, 100.0f, 200.0f, 99999.f}, "Mixing bins - multiplicity"};

  std::vector<float> CastCfgVtxBins, CastCfgMultBins;

  Produces<aod::Hashes> hashes;

  void init(InitContext&)
  {
    /// here the Configurables are passed to std::vectors
    CastCfgVtxBins = (std::vector<float>)CfgVtxBins;
    CastCfgMultBins = (std::vector<float>)CfgMultBins;
  }

  void process(o2::aod::FemtoDreamCollision const& col)
  {
    /// the hash of the collision is computed and written to table
    hashes(eventmixing::getMixingBin(CastCfgVtxBins, CastCfgMultBins, col.posZ(), col.multV0M()));
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{
    adaptAnalysisTask<femtoDreamPairHashTask>(cfgc)};

  return workflow;
}
