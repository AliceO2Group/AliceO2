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

/// \file TRDPIDSpec.cxx
/// \brief This file provides the specification for calculating the pid value.
/// \author Felix Schlepper

#include "TRDWorkflow/TRDPIDSpec.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "CommonDataFormat/IRFrame.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

#include <gsl/span>

using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

void TRDPIDSpec::init(o2::framework::InitContext& ic)
{
}

void TRDPIDSpec::run(o2::framework::ProcessingContext& pc)
{
}

o2::framework::DataProcessorSpec getTRDPIDSpec(o2::dataformats::GlobalTrackID::mask_t src)
{
  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();

  GTrackID::mask_t srcClu = GTrackID::getSourcesMask("TRD");
  dataRequest->requestTracks(src, false);      // Request ITS-TPC-TRD and TPC-TRD tracks
  dataRequest->requestClusters(srcClu, false); // Cluster = tracklets for trd
  auto& inputs = dataRequest->inputs;
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              false,                             // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              inputs,
                                                              true);
  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "TRDPID",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDPIDSpec>(dataRequest, ggRequest)},
    Options{}};
}

} // end namespace trd
} // end namespace o2
