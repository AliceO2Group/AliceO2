// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_DCS_DATAGENERATOR_H
#define O2_DCS_DATAGENERATOR_H

/// @file   DataGeneratorSpec.h
/// @brief  Dummy data generator
#include <unistd.h>
#include <TRandom.h>
#include <TDatime.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"
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
class DCSDataGenerator : public o2::framework::Task
{

  using DPID = o2::dcs::DataPointIdentifier;
  using DPVAL = o2::dcs::DataPointValue;

 public:
  void init(o2::framework::InitContext& ic) final
  {
    mMaxTF = ic.options().get<int64_t>("max-timeframes");

    LOG(INFO) << "mMaxTF = " << mMaxTF;
    std::string dpAliaschar = "TestChar0";
    DPID::FILL(mcharVar, dpAliaschar, mtypechar);

    std::string dpAliasint0 = "TestInt0";
    DPID::FILL(mintVar0, dpAliasint0, mtypeint);
    std::string dpAliasint1 = "TestInt1";
    DPID::FILL(mintVar1, dpAliasint1, mtypeint);
    std::string dpAliasint2 = "TestInt2";
    DPID::FILL(mintVar2, dpAliasint2, mtypeint);

    std::string dpAliasdouble0 = "TestDouble0";
    DPID::FILL(mdoubleVar0, dpAliasdouble0, mtypedouble);
    std::string dpAliasdouble1 = "TestDouble1";
    DPID::FILL(mdoubleVar1, dpAliasdouble1, mtypedouble);
    std::string dpAliasdouble2 = "TestDouble2";
    DPID::FILL(mdoubleVar2, dpAliasdouble2, mtypedouble);
    std::string dpAliasdouble3 = "TestDouble3";
    DPID::FILL(mdoubleVar3, dpAliasdouble3, mtypedouble);

    std::string dpAliasstring0 = "TestString0";
    DPID::FILL(mstringVar0, dpAliasstring0, mtypestring);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {

    uint64_t tfid;
    for (auto& input : pc.inputs()) {
      tfid = header::get<o2::framework::DataProcessingHeader*>(input.header)->startTime;
      LOG(INFO) << "tfid = " << tfid;
      if (tfid >= mMaxTF) {
        LOG(INFO) << "Data generator reached TF " << tfid << ", stopping";
        pc.services().get<o2::framework::ControlService>().endOfStream();
        pc.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me);
        break;
      }
    }

    uint16_t flags = 0;
    uint16_t milliseconds = 0;
    TDatime currentTime;
    uint32_t seconds = currentTime.Get();
    uint64_t* payload = new uint64_t[7];
    memset(payload, 0, sizeof(uint64_t) * 7);

    payload[0] = (uint64_t)tfid + 33; // adding 33 to have visible chars and strings

    DPVAL valchar(flags, milliseconds + tfid * 10, seconds + tfid, payload, mtypechar);
    DPVAL valint(flags, milliseconds + tfid * 10, seconds + tfid, payload, mtypeint);
    DPVAL valdouble(flags, milliseconds + tfid * 10, seconds + tfid, payload, mtypedouble);
    DPVAL valstring(flags, milliseconds + tfid * 10, seconds + tfid, payload, mtypestring);

    LOG(DEBUG) << mcharVar;
    LOG(DEBUG) << valchar << " --> " << (char)valchar.payload_pt1;
    LOG(DEBUG) << mintVar0;
    LOG(DEBUG) << valint << " --> " << (int)valint.payload_pt1;
    LOG(DEBUG) << mintVar1;
    LOG(DEBUG) << valint << " --> " << (int)valint.payload_pt1;
    LOG(DEBUG) << mintVar2;
    LOG(DEBUG) << valint << " --> " << (int)valint.payload_pt1;
    LOG(DEBUG) << mdoubleVar0;
    LOG(DEBUG) << valdouble << " --> " << (double)valdouble.payload_pt1;
    LOG(DEBUG) << mdoubleVar1;
    LOG(DEBUG) << valdouble << " --> " << (double)valdouble.payload_pt1;
    LOG(DEBUG) << mdoubleVar2;
    LOG(DEBUG) << valdouble << " --> " << (double)valdouble.payload_pt1;
    LOG(DEBUG) << mdoubleVar3;
    LOG(DEBUG) << valdouble << " --> " << (double)valdouble.payload_pt1;
    char tt[56];
    memcpy(&tt[0], &valstring.payload_pt1, 56);
    LOG(DEBUG) << mstringVar0;
    LOG(DEBUG) << valstring << " --> " << tt;
    auto& tmpDPmap = pc.outputs().make<std::unordered_map<DPID, DPVAL>>(o2::framework::OutputRef{"output", 0});
    tmpDPmap[mcharVar] = valchar;
    tmpDPmap[mintVar0] = valint;
    tmpDPmap[mintVar1] = valint;
    tmpDPmap[mintVar2] = valint;
    tmpDPmap[mdoubleVar0] = valdouble;
    tmpDPmap[mdoubleVar1] = valdouble;
    tmpDPmap[mdoubleVar2] = valdouble;
    if (tfid % 3 == 0)
      tmpDPmap[mdoubleVar3] = valdouble; // to test the case when a DP is not updated, we skip some updates
    tmpDPmap[mstringVar0] = valstring;
    delete payload;
  }

 private:
  uint64_t mMaxTF = 1;
  o2::dcs::DataPointIdentifier mcharVar;
  DPID mintVar0;
  DPID mintVar1;
  DPID mintVar2;
  DPID mdoubleVar0;
  DPID mdoubleVar1;
  DPID mdoubleVar2;
  DPID mdoubleVar3;
  DPID mstringVar0;
  DeliveryType mtypechar = RAW_CHAR;
  DeliveryType mtypeint = RAW_INT;
  DeliveryType mtypedouble = RAW_DOUBLE;
  DeliveryType mtypestring = RAW_STRING;
};

} // namespace dcs

namespace framework
{

DataProcessorSpec getDCSDataGeneratorSpec()
{
  return DataProcessorSpec{
    "dcs-data-generator",
    Inputs{},
    Outputs{{{"output"}, "DCS", "DATAPOINTS"}},
    AlgorithmSpec{adaptFromTask<o2::dcs::DCSDataGenerator>()},
    Options{{"max-timeframes", VariantType::Int64, 99999999999ll, {"max TimeFrames to generate"}}}};
}

} // namespace framework
} // namespace o2

#endif
