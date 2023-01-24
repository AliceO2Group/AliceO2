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

/// \file TOFIntegrateClusterSpec.cxx
/// \brief device for integrating the TOF clusters in time slices
/// \author Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Jan 21, 2023

#include "TOFWorkflowUtils/TOFIntegrateClusterSpec.h"
#include "Framework/Task.h"
#include "DataFormatsTOF/Cluster.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TOFBase/Geo.h"
#include "DetectorsBase/TFIDInfoHelper.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

class TOFIntegrateClusters : public Task
{
 public:
  /// \constructor
  TOFIntegrateClusters(std::shared_ptr<o2::base::GRPGeomRequest> req, const bool disableWriter) : mCCDBRequest(req), mDisableWriter(disableWriter){};

  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    mNSlicesTF = ic.options().get<int>("nSlicesTF");
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    const int nHBFPerTF = o2::base::GRPGeomHelper::instance().getNHBFPerTF();
    const double orbitLength = o2::tof::Geo::BC_TIME_INPS * o2::constants::lhc::LHCMaxBunches;
    const double maxTimeTF = orbitLength * nHBFPerTF;   // maximum time if a TF in PS
    const double sliceWidthPS = maxTimeTF / mNSlicesTF; // integration time
    const double sliceWidthMS = sliceWidthPS * 1.E-9;
    const int nSlices = maxTimeTF / sliceWidthPS; // number of slices a TF is divided into
    const double sliceWidthPSinv = 1. / sliceWidthPS;
    const float sliceWidthMSinv = 1. / float(sliceWidthMS);

    // storage for integrated currents
    o2::pmr::vector<float> iTOFCNCl(nSlices);
    o2::pmr::vector<float> iTOFCqTot(nSlices);

    const auto clusters = pc.inputs().get<gsl::span<o2::tof::Cluster>>("tofcluster");
    for (size_t iCl = 0; iCl < clusters.size(); ++iCl) {
      const double timePS = clusters[iCl].getTime();
      const unsigned int sliceInTF = timePS * sliceWidthPSinv;
      if (sliceInTF < mNSlicesTF) {
        ++iTOFCNCl[sliceInTF];                          // count number of cluster
        iTOFCqTot[sliceInTF] += clusters[iCl].getTot(); // integrated charge
      } else {
        LOGP(info, "slice in TF of ICC {} is larger than max slice {} with nTSPerSlice {}", sliceInTF, mNSlicesTF, nSlices);
      }
    }

    // normalize currents to integration time
    std::transform(iTOFCNCl.begin(), iTOFCNCl.end(), iTOFCNCl.begin(), [sliceWidthMSinv](float& val) { return val * sliceWidthMSinv; });
    std::transform(iTOFCqTot.begin(), iTOFCqTot.end(), iTOFCqTot.begin(), [sliceWidthMSinv](float& val) { return val * sliceWidthMSinv; });

    sendOutput(pc, std::move(iTOFCNCl), std::move(iTOFCqTot));
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    LOGP(info, "Finalizing calibration");
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final { o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj); }

 private:
  int mNSlicesTF = 11;                                    ///< number of slices a TF is divided into
  const bool mDisableWriter{false};                       ///< flag if no ROOT output will be written
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest; ///< info for CCDB request

  void sendOutput(ProcessingContext& pc, o2::pmr::vector<float> iTOFCNCl, o2::pmr::vector<float> iTOFCqTot)
  {
    pc.outputs().adoptContainer(Output{header::gDataOriginTOF, "ITOFCN"}, std::move(iTOFCNCl));
    pc.outputs().adoptContainer(Output{header::gDataOriginTOF, "ITOFCQ"}, std::move(iTOFCqTot));
    // in case of ROOT output also store the TFinfo in the TTree
    if (!mDisableWriter) {
      o2::dataformats::TFIDInfo tfinfo;
      o2::base::TFIDInfoHelper::fillTFIDInfo(pc, tfinfo);
      pc.outputs().snapshot(Output{header::gDataOriginTOF, "ITOFTFID"}, tfinfo);
    }
  }
};

o2::framework::DataProcessorSpec getTOFIntegrateClusterSpec(const bool disableWriter)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("tofcluster", o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe);

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                                true,                           // GRPECS=true for nHBF per TF
                                                                false,                          // GRPLHCIF
                                                                false,                          // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "ITOFCN", 0, Lifetime::Sporadic);
  outputs.emplace_back(o2::header::gDataOriginTOF, "ITOFCQ", 0, Lifetime::Sporadic);
  if (!disableWriter) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "ITOFTFID", 0, Lifetime::Sporadic);
  }

  return DataProcessorSpec{
    "TOFIntegrateClusters",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TOFIntegrateClusters>(ccdbRequest, disableWriter)},
    Options{{"nSlicesTF", VariantType::Int, 11, {"number of slices into which a TF is divided"}}}};
}

} // end namespace tof
} // end namespace o2
