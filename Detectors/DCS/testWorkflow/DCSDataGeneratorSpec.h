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
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"

using namespace o2::framework;

namespace o2
{
namespace dcs
{
class DCSDataGenerator : public o2::framework::Task
{

  using DPID = o2::dcs::DataPointIdentifier;
  using DPVAL = o2::dcs::DataPointValue;
  using DPCOM = o2::dcs::DataPointCompositeObject;

 public:
  void init(o2::framework::InitContext& ic) final
  {
    mMaxTF = ic.options().get<int64_t>("max-timeframes");

    LOG(INFO) << "mMaxTF = " << mMaxTF;

    DPID dpidtmp;

    // chars
    std::string dpAliaschar = "TestChar_0";
    DPID::FILL(dpidtmp, dpAliaschar, mtypechar);
    mDPIDvect.push_back(dpidtmp);
    mNumDPs++;
    mNumDPschar++;

    // ints
    for (int i = 0; i < 3; i++) {
      std::string dpAliasint = "TestInt_" + std::to_string(i);
      DPID::FILL(dpidtmp, dpAliasint, mtypeint);
      mDPIDvect.push_back(dpidtmp);
      mNumDPs++;
      mNumDPsint++;
    }

    // doubles
    for (int i = 0; i < 4; i++) {
      std::string dpAliasdouble = "TestDouble_" + std::to_string(i);
      DPID::FILL(dpidtmp, dpAliasdouble, mtypedouble);
      mDPIDvect.push_back(dpidtmp);
      mNumDPs++;
      mNumDPsdouble++;
    }

    // strings
    std::string dpAliasstring0 = "TestString_0";
    DPID::FILL(dpidtmp, dpAliasstring0, mtypestring);
    mDPIDvect.push_back(dpidtmp);
    mNumDPs++;
    mNumDPsstring++;

    LOG(INFO) << "Number of DCS data points = " << mNumDPs;
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
      }
      break; // we break because one input is enough to get the TF ID
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

    LOG(INFO) << "Value used for char DPs:";
    LOG(INFO) << valchar << " --> " << (char)valchar.payload_pt1;
    LOG(INFO) << "Value used for int DPs:";
    LOG(INFO) << valint << " --> " << (int)valint.payload_pt1;
    LOG(INFO) << "Value used for double DPs:";
    LOG(INFO) << valdouble << " --> " << (double)valdouble.payload_pt1;
    char tt[56];
    memcpy(&tt[0], &valstring.payload_pt1, 56);
    LOG(INFO) << "Value used for string DPs:";
    LOG(INFO) << valstring << " --> " << tt;

    std::vector<DPCOM> dpcomVect;
    for (int i = 0; i < mNumDPschar; i++) {
      dpcomVect.emplace_back(mDPIDvect[i], valchar);
    }
    for (int i = 0; i < mNumDPsint; i++) {
      dpcomVect.emplace_back(mDPIDvect[mNumDPschar + i], valint);
    }
    for (int i = 0; i < mNumDPsdouble; i++) {
      dpcomVect.emplace_back(mDPIDvect[mNumDPschar + mNumDPsint + i], valdouble);
    }
    for (int i = 0; i < mNumDPsstring; i++) {
      dpcomVect.emplace_back(mDPIDvect[mNumDPschar + mNumDPsint + mNumDPsdouble + i], valstring);
    }

    auto svect = dpcomVect.size();
    LOG(INFO) << "dpcomVect has size " << svect;
    for (int i = 0; i < svect; i++) {
      LOG(INFO) << "i = " << i << ", DPCOM = " << dpcomVect[i];
    }
    std::vector<char> buff(mNumDPs * sizeof(DPCOM));
    char* dptr = buff.data();
    for (int i = 0; i < svect; i++) {
      memcpy(dptr + i * sizeof(DPCOM), &dpcomVect[i], sizeof(DPCOM));
    }
    auto sbuff = buff.size();
    LOG(INFO) << "size of output buffer = " << sbuff;
    pc.outputs().snapshot(Output{"DCS", "DATAPOINTS", 0, Lifetime::Timeframe}, buff.data(), sbuff);

    LOG(INFO) << "Reading back";
    DPCOM dptmp;
    for (int i = 0; i < svect; i++) {
      memcpy(&dptmp, dptr + i * sizeof(DPCOM), sizeof(DPCOM));
      LOG(INFO) << "Check: Reading from generator: i = " << i << ", DPCOM = " << dptmp;
    }
    /*
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
    */
    delete payload;
  }

 private:
  uint64_t mMaxTF = 1;
  uint64_t mNumDPs = 0;
  uint64_t mNumDPschar = 0;
  uint64_t mNumDPsint = 0;
  uint64_t mNumDPsdouble = 0;
  uint64_t mNumDPsstring = 0;
  /*
  DPID mcharVar;
  DPID mintVar0;
  DPID mintVar1;
  DPID mintVar2;
  DPID mdoubleVar0;
  DPID mdoubleVar1;
  DPID mdoubleVar2;
  DPID mdoubleVar3;
  DPID mstringVar0;
  */
  std::vector<DPID> mDPIDvect;
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
    Outputs{{{"outputDCS"}, "DCS", "DATAPOINTS"}},
    AlgorithmSpec{adaptFromTask<o2::dcs::DCSDataGenerator>()},
    Options{{"max-timeframes", VariantType::Int64, 99999999999ll, {"max TimeFrames to generate"}}}};
}

} // namespace framework
} // namespace o2

#endif
