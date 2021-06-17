// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   TrackBasedCalibSpec.cxx
/// \brief DPL device for creating/providing track based TRD calibration input
/// \author Ole Schmidt

#include "TRDWorkflow/TrackBasedCalibSpec.h"
#include "TRDCalibration/TrackBasedCalib.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Headers/DataHeader.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

class TRDTrackBasedCalibDevice : public Task
{
 public:
  TRDTrackBasedCalibDevice(std::shared_ptr<DataRequest> dr) : mDataRequest(dr) {}
  ~TRDTrackBasedCalibDevice() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  std::shared_ptr<DataRequest> mDataRequest;
  o2::trd::TrackBasedCalib mCalibrator; // gather input data for calibration of vD, ExB and gain
};

void TRDTrackBasedCalibDevice::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};
  mCalibrator.init();
}

void TRDTrackBasedCalibDevice::run(ProcessingContext& pc)
{

  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());

  mCalibrator.calculateAngResHistos(recoData);

  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "ANGRESHISTS", 0, Lifetime::Timeframe}, mCalibrator.getAngResHistos());
}

void TRDTrackBasedCalibDevice::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "Added in total %i entries to angular residual histograms",
       mCalibrator.getAngResHistos().getNEntries());
}

DataProcessorSpec getTRDTrackBasedCalibSpec()
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  GTrackID::mask_t srcTrk = GTrackID::getSourcesMask("ITS-TPC-TRD"); // possibly also use TPC-TRD?
  GTrackID::mask_t srcClu = GTrackID::getSourcesMask("TRD");         // we don't need all clusters, only TRD tracklets
  dataRequest->requestTracks(srcTrk, false);
  dataRequest->requestClusters(srcClu, false);

  outputs.emplace_back(o2::header::gDataOriginTRD, "ANGRESHISTS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "trd-trackbased-calib",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDTrackBasedCalibDevice>(dataRequest)},
    Options{}};
}

} // namespace trd
} // namespace o2
