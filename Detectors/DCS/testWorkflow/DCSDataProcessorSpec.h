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
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/DCSProcessor.h"
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

class DCSDataProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    std::vector<DPID> aliasVect;

    DeliveryType typechar = RAW_CHAR;
    std::string dpAliaschar = "TestChar0";
    DPID charVar(dpAliaschar, typechar);
    aliasVect.push_back(charVar);

    DeliveryType typeint = RAW_INT;
    std::string dpAliasint0 = "TestInt0";
    DPID intVar0(dpAliasint0, typeint);
    aliasVect.push_back(intVar0);
    std::string dpAliasint1 = "TestInt1";
    DPID intVar1(dpAliasint1, typeint);
    aliasVect.push_back(intVar1);
    std::string dpAliasint2 = "TestInt2";
    DPID intVar2(dpAliasint2, typeint);
    aliasVect.push_back(intVar2);

    DeliveryType typedouble = RAW_DOUBLE;
    std::string dpAliasdouble0 = "TestDouble0";
    DPID doubleVar0(dpAliasdouble0, typedouble);
    aliasVect.push_back(doubleVar0);
    std::string dpAliasdouble1 = "TestDouble1";
    DPID doubleVar1(dpAliasdouble1, typedouble);
    aliasVect.push_back(doubleVar1);
    std::string dpAliasdouble2 = "TestDouble2";
    DPID doubleVar2(dpAliasdouble2, typedouble);
    aliasVect.push_back(doubleVar2);
    std::string dpAliasdouble3 = "TestDouble3";
    DPID doubleVar3(dpAliasdouble3, typedouble);
    aliasVect.push_back(doubleVar3);

    DeliveryType typestring = RAW_STRING;
    std::string dpAliasstring0 = "TestString0";
    DPID stringVar0(dpAliasstring0, typestring);
    aliasVect.push_back(stringVar0);

    mDCSproc.init(aliasVect);
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    auto dcsmap = pc.inputs().get<std::unordered_map<DPID, DPVAL>*>("input");
    mDCSproc.process(*dcsmap);
  }

 private:
  o2::dcs::DCSProcessor mDCSproc;
};

} // namespace dcs

namespace framework
{

DataProcessorSpec getDCSDataProcessorSpec()
{
  return DataProcessorSpec{
    "dcs-data-processor",
    Inputs{{"input", "DCS", "DATAPOINTS"}},
    Outputs{{}},
    AlgorithmSpec{adaptFromTask<o2::dcs::DCSDataProcessor>()},
    Options{}};
}

} // namespace framework
} // namespace o2

#endif
