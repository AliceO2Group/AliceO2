// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_DCS_DATAPROCESSOR_H
#define O2_DCS_DATAPROCESSOR_H

/// @file   DataGeneratorSpec.h
/// @brief  Dummy data generator

#include <unistd.h>
#include <TRandom.h>
#include <TStopwatch.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DCSProcessor.h"
#include "DetectorsCalibration/Utils.h"
#include "CCDB/CcdbApi.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

namespace o2
{
namespace dcs
{

using namespace o2::dcs;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

class DCSDataProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    std::vector<DPID> aliasVect;

    DPID dpidtmp;
    DeliveryType typechar = RAW_CHAR;
    std::string dpAliaschar = "TestChar_0";
    DPID::FILL(dpidtmp, dpAliaschar, typechar);
    aliasVect.push_back(dpidtmp);

    DeliveryType typeint = RAW_INT;
    for (int i = 0; i < 50000; i++) {
      std::string dpAliasint = "TestInt_" + std::to_string(i);
      DPID::FILL(dpidtmp, dpAliasint, typeint);
      aliasVect.push_back(dpidtmp);
    }

    DeliveryType typedouble = RAW_DOUBLE;
    for (int i = 0; i < 4; i++) {
      std::string dpAliasdouble = "TestDouble_" + std::to_string(i);
      DPID::FILL(dpidtmp, dpAliasdouble, typedouble);
      aliasVect.push_back(dpidtmp);
    }

    DeliveryType typestring = RAW_STRING;
    std::string dpAliasstring0 = "TestString_0";
    DPID::FILL(dpidtmp, dpAliasstring0, typestring);
    aliasVect.push_back(dpidtmp);

    mDCSproc.init(aliasVect);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfid = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    mDCSproc.setTF(tfid);

    TStopwatch s;
    LOG(INFO) << "TF: " << tfid << " -->  receiving binary data...";
    s.Start();
    auto rawchar = pc.inputs().get<const char*>("input");
    s.Stop();
    LOG(INFO) << "TF: " << tfid << " -->  ...binary data received: realTime = " << s.RealTime() << ", cpuTime = " << s.CpuTime();
    LOG(INFO) << "TF: " << tfid << " -->  receiving (delta) binary data...";
    s.Start();
    auto rawcharDelta = pc.inputs().get<const char*>("inputDelta");
    s.Stop();
    LOG(INFO) << "TF: " << tfid << " -->  ...binary (delta) data received: realTime = " << s.RealTime() << ", cpuTime = " << s.CpuTime();

    // full map
    const auto* dh = o2::header::get<o2::header::DataHeader*>(pc.inputs().get("input").header);
    auto sz = dh->payloadSize;
    int nDPs = sz / sizeof(DPCOM);
    LOG(INFO) << "Number of DPs received = " << nDPs;
    std::unordered_map<DPID, DPVAL> dcsmap;
    DPCOM dptmp;
    LOG(INFO) << "TF: " << tfid << " -->  building unordered_map...";
    s.Start();
    for (int i = 0; i < nDPs; i++) {
      memcpy(&dptmp, rawchar + i * sizeof(DPCOM), sizeof(DPCOM));
      dcsmap[dptmp.id] = dptmp.data;
      LOG(DEBUG) << "Reading from generator: i = " << i << ", DPCOM = " << dptmp;
      LOG(DEBUG) << "Reading from generator: i = " << i << ", DPID = " << dptmp.id;
    }
    s.Stop();
    LOG(INFO) << "TF: " << tfid << " -->  ...unordered_map built = " << s.RealTime() << ", cpuTime = " << s.CpuTime();

    // delta map
    const auto* dhDelta = o2::header::get<o2::header::DataHeader*>(pc.inputs().get("inputDelta").header);
    auto szDelta = dhDelta->payloadSize;
    int nDPsDelta = szDelta / sizeof(DPCOM);
    LOG(INFO) << "Number of DPs received (delta map) = " << nDPsDelta;
    std::unordered_map<DPID, DPVAL> dcsmapDelta;
    LOG(INFO) << "TF: " << tfid << " -->  building (delta) unordered_map...";
    s.Start();
    for (int i = 0; i < nDPsDelta; i++) {
      memcpy(&dptmp, rawcharDelta + i * sizeof(DPCOM), sizeof(DPCOM));
      dcsmapDelta[dptmp.id] = dptmp.data;
      LOG(DEBUG) << "Reading from generator: i = " << i << ", DPCOM = " << dptmp;
      LOG(DEBUG) << "Reading from generator: i = " << i << ", DPID = " << dptmp.id;
    }
    s.Stop();
    LOG(INFO) << "TF: " << tfid << " -->  ...unordered_map (delta) built = " << s.RealTime() << ", cpuTime = " << s.CpuTime();

    LOG(INFO) << "TF: " << tfid << " -->  starting processing...";
    s.Start();
    mDCSproc.process(dcsmap, false);
    s.Stop();
    LOG(INFO) << "TF: " << tfid << " -->  ...processing done: realTime = " << s.RealTime() << ", cpuTime = " << s.CpuTime();

    LOG(INFO) << "TF: " << tfid << " -->  starting (delta) processing...";
    s.Start();
    mDCSproc.process(dcsmapDelta, true);
    s.Stop();
    LOG(INFO) << "TF: " << tfid << " -->  ...processing (delta) done: realTime = " << s.RealTime() << ", cpuTime = " << s.CpuTime();

    sendOutput(pc.outputs());
  }

 private:
  o2::dcs::DCSProcessor mDCSproc;

  //________________________________________________________________
  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
    // copied from LHCClockCalibratorSpec.cxx
    using clbUtils = o2::calibration::Utils;
    const auto& payload = mDCSproc.getCCDBint();
    auto& info = mDCSproc.getCCDBintInfo();
    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size() << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

    output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload, 0}, *image.get()); // vector<char>
    output.snapshot(Output{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo, 0}, info);
  }
}; // end class
} // namespace dcs

namespace framework
{

DataProcessorSpec getDCSDataProcessorSpec()
{

  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBPayload});
  outputs.emplace_back(ConcreteDataTypeMatcher{clbUtils::gDataOriginCLB, clbUtils::gDataDescriptionCLBInfo});

  return DataProcessorSpec{
    "dcs-data-processor",
    Inputs{{"input", "DCS", "DATAPOINTS"}, {"inputDelta", "DCS", "DATAPOINTSdelta"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::dcs::DCSDataProcessor>()},
    Options{}};
}

} // namespace framework
} // namespace o2

#endif
