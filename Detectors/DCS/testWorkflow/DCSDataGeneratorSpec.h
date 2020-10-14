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
#include <TStopwatch.h>
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
    mDPIDvectFull.push_back(dpidtmp);
    mNumDPsFull++;
    mNumDPscharFull++;

    // ints
    for (int i = 0; i < 50000; i++) {
      std::string dpAliasint = "TestInt_" + std::to_string(i);
      DPID::FILL(dpidtmp, dpAliasint, mtypeint);
      mDPIDvectFull.push_back(dpidtmp);
      mNumDPsFull++;
      mNumDPsintFull++;
      if (i < 100) {
        mDPIDvectDelta.push_back(dpidtmp);
        mNumDPsDelta++;
        mNumDPsintDelta++;
      }
    }

    // doubles
    for (int i = 0; i < 4; i++) {
      std::string dpAliasdouble = "TestDouble_" + std::to_string(i);
      DPID::FILL(dpidtmp, dpAliasdouble, mtypedouble);
      mDPIDvectFull.push_back(dpidtmp);
      mNumDPsFull++;
      mNumDPsdoubleFull++;
    }

    // strings
    std::string dpAliasstring0 = "TestString_0";
    DPID::FILL(dpidtmp, dpAliasstring0, mtypestring);
    mDPIDvectFull.push_back(dpidtmp);
    mNumDPsFull++;
    mNumDPsstringFull++;

    LOG(INFO) << "Number of DCS data points: " << mNumDPsFull << " (full map); " << mNumDPsDelta << " (delta map)";
  }

  void run(o2::framework::ProcessingContext& pc) final
  {

    uint64_t tfid;
    for (auto& input : pc.inputs()) {
      tfid = header::get<o2::framework::DataProcessingHeader*>(input.header)->startTime;
      LOG(DEBUG) << "tfid = " << tfid;
      if (tfid >= mMaxTF) {
        LOG(INFO) << "Data generator reached TF " << tfid << ", stopping";
        pc.services().get<o2::framework::ControlService>().endOfStream();
        pc.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me);
      }
      break; // we break because one input is enough to get the TF ID
    }

    LOG(DEBUG) << "TF: " << tfid << " --> building binary blob...";
    uint16_t flags = 0;
    uint16_t milliseconds = 0;
    TDatime currentTime;
    uint32_t seconds = currentTime.Get();
    uint64_t payload[7];
    memset(payload, 0, sizeof(uint64_t) * 7);

    payload[0] = (uint64_t)tfid + 33; // adding 33 to have visible chars and strings

    DPVAL valchar(flags, milliseconds + tfid * 10, seconds + tfid, payload, mtypechar);
    DPVAL valint(flags, milliseconds + tfid * 10, seconds + tfid, payload, mtypeint);
    DPVAL valdouble(flags, milliseconds + tfid * 10, seconds + tfid, payload, mtypedouble);
    DPVAL valstring(flags, milliseconds + tfid * 10, seconds + tfid, payload, mtypestring);

    LOG(DEBUG) << "Value used for char DPs:";
    LOG(DEBUG) << valchar << " --> " << (char)valchar.payload_pt1;
    LOG(DEBUG) << "Value used for int DPs:";
    LOG(DEBUG) << valint << " --> " << (int)valint.payload_pt1;
    LOG(DEBUG) << "Value used for double DPs:";
    LOG(DEBUG) << valdouble << " --> " << (double)valdouble.payload_pt1;
    char tt[56];
    memcpy(&tt[0], &valstring.payload_pt1, 56);
    LOG(DEBUG) << "Value used for string DPs:";
    LOG(DEBUG) << valstring << " --> " << tt;

    // full map (all DPs)
    mBuildingBinaryBlock.Start(mFirstTF);
    std::vector<DPCOM> dpcomVectFull;
    for (int i = 0; i < mNumDPscharFull; i++) {
      dpcomVectFull.emplace_back(mDPIDvectFull[i], valchar);
    }
    for (int i = 0; i < mNumDPsintFull; i++) {
      dpcomVectFull.emplace_back(mDPIDvectFull[mNumDPscharFull + i], valint);
    }
    for (int i = 0; i < mNumDPsdoubleFull; i++) {
      dpcomVectFull.emplace_back(mDPIDvectFull[mNumDPscharFull + mNumDPsintFull + i], valdouble);
    }
    for (int i = 0; i < mNumDPsstringFull; i++) {
      dpcomVectFull.emplace_back(mDPIDvectFull[mNumDPscharFull + mNumDPsintFull + mNumDPsdoubleFull + i], valstring);
    }

    // delta map (only DPs that changed)
    mDeltaBuildingBinaryBlock.Start(mFirstTF);
    std::vector<DPCOM> dpcomVectDelta;
    for (int i = 0; i < mNumDPscharDelta; i++) {
      dpcomVectDelta.emplace_back(mDPIDvectDelta[i], valchar);
    }
    for (int i = 0; i < mNumDPsintDelta; i++) {
      dpcomVectDelta.emplace_back(mDPIDvectDelta[mNumDPscharDelta + i], valint);
    }
    for (int i = 0; i < mNumDPsdoubleDelta; i++) {
      dpcomVectDelta.emplace_back(mDPIDvectDelta[mNumDPscharDelta + mNumDPsintDelta + i], valdouble);
    }
    for (int i = 0; i < mNumDPsstringDelta; i++) {
      dpcomVectDelta.emplace_back(mDPIDvectDelta[mNumDPscharDelta + mNumDPsintDelta + mNumDPsdoubleDelta + i],
                                  valstring);
    }

    // Full map
    auto svect = dpcomVectFull.size();
    LOG(DEBUG) << "dpcomVectFull has size " << svect;
    for (int i = 0; i < svect; i++) {
      LOG(DEBUG) << "i = " << i << ", DPCOM = " << dpcomVectFull[i];
    }
    std::vector<char> buff(mNumDPsFull * sizeof(DPCOM));
    char* dptr = buff.data();
    for (int i = 0; i < svect; i++) {
      memcpy(dptr + i * sizeof(DPCOM), &dpcomVectFull[i], sizeof(DPCOM));
    }
    auto sbuff = buff.size();
    LOG(DEBUG) << "size of output buffer = " << sbuff;
    mBuildingBinaryBlock.Stop();
    LOG(DEBUG) << "TF: " << tfid << " --> ...binary blob prepared: realTime = "
               << mBuildingBinaryBlock.RealTime() << ", cpuTime = "
               << mBuildingBinaryBlock.CpuTime();
    LOG(DEBUG) << "TF: " << tfid << " --> sending snapshot...";
    mSnapshotSending.Start(mFirstTF);
    pc.outputs().snapshot(Output{"DCS", "DATAPOINTS", 0, Lifetime::Timeframe}, buff.data(), sbuff);
    mSnapshotSending.Stop();
    LOG(DEBUG) << "TF: " << tfid << " --> ...snapshot sent: realTime = " << mSnapshotSending.RealTime()
               << ", cpuTime = " << mSnapshotSending.CpuTime();

    // Delta map
    auto svectDelta = dpcomVectDelta.size();
    LOG(DEBUG) << "dpcomVectDelta has size " << svect;
    for (int i = 0; i < svectDelta; i++) {
      LOG(DEBUG) << "i = " << i << ", DPCOM = " << dpcomVectDelta[i];
    }
    std::vector<char> buffDelta(mNumDPsDelta * sizeof(DPCOM));
    char* dptrDelta = buffDelta.data();
    for (int i = 0; i < svectDelta; i++) {
      memcpy(dptrDelta + i * sizeof(DPCOM), &dpcomVectDelta[i], sizeof(DPCOM));
    }
    auto sbuffDelta = buffDelta.size();
    LOG(DEBUG) << "size of output (delta) buffer = " << sbuffDelta;
    mDeltaBuildingBinaryBlock.Stop();
    LOG(DEBUG) << "TF: " << tfid << " --> ...binary (delta) blob prepared: realTime = "
               << mDeltaBuildingBinaryBlock.RealTime() << ", cpuTime = " << mDeltaBuildingBinaryBlock.CpuTime();
    LOG(DEBUG) << "TF: " << tfid << " --> sending (delta) snapshot...";
    mDeltaSnapshotSending.Start(mFirstTF);
    pc.outputs().snapshot(Output{"DCS", "DATAPOINTSdelta", 0, Lifetime::Timeframe}, buffDelta.data(), sbuffDelta);
    mDeltaSnapshotSending.Stop();
    LOG(DEBUG) << "TF: " << tfid << " --> ...snapshot (delta) sent: realTime = "
               << mDeltaSnapshotSending.RealTime() << ", cpuTime = "
               << mDeltaSnapshotSending.CpuTime();

    /*
    LOG(INFO) << "Reading back";
    DPCOM dptmp;
    for (int i = 0; i < svect; i++) {
      memcpy(&dptmp, dptr + i * sizeof(DPCOM), sizeof(DPCOM));
      LOG(DEBUG) << "Check: Reading from generator: i = " << i << ", DPCOM = " << dptmp;
    }
    */
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
    mFirstTF = false;
    mTFs++;
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOG(INFO) << "number of processed TF: " << mTFs;
    LOG(INFO) << " --> time to prepare binary blob: realTime = "
              << mBuildingBinaryBlock.RealTime() / mTFs << ", cpuTime = "
              << mBuildingBinaryBlock.CpuTime() / mTFs;
    LOG(INFO) << " --> time to send snapshot: realTime = "
              << mSnapshotSending.RealTime() / mTFs << ", cpuTime = "
              << mSnapshotSending.CpuTime() / mTFs;
    LOG(INFO) << " --> time to prepare binary blob: realTime = "
              << mDeltaBuildingBinaryBlock.RealTime() / mTFs << ", cpuTime = "
              << mDeltaBuildingBinaryBlock.CpuTime() / mTFs;
    LOG(INFO) << " --> time to send snapshot: realTime = "
              << mDeltaSnapshotSending.RealTime() / mTFs << ", cpuTime = "
              << mDeltaSnapshotSending.CpuTime() / mTFs;
  }

 private:
  uint64_t mMaxTF = 1;

  uint64_t mNumDPsFull = 0;
  uint64_t mNumDPscharFull = 0;
  uint64_t mNumDPsintFull = 0;
  uint64_t mNumDPsdoubleFull = 0;
  uint64_t mNumDPsstringFull = 0;

  uint64_t mNumDPsDelta = 0;
  uint64_t mNumDPscharDelta = 0;
  uint64_t mNumDPsintDelta = 0;
  uint64_t mNumDPsdoubleDelta = 0;
  uint64_t mNumDPsstringDelta = 0;
  std::vector<DPID> mDPIDvectFull;  // for full map
  std::vector<DPID> mDPIDvectDelta; // for delta map (containing only DPs that changed)
  DeliveryType mtypechar = RAW_CHAR;
  DeliveryType mtypeint = RAW_INT;
  DeliveryType mtypedouble = RAW_DOUBLE;
  DeliveryType mtypestring = RAW_STRING;

  TStopwatch mBuildingBinaryBlock;
  TStopwatch mDeltaBuildingBinaryBlock;
  TStopwatch mSnapshotSending;
  TStopwatch mDeltaSnapshotSending;
  bool mFirstTF = true;
  uint64_t mTFs = 0;
};

} // namespace dcs

namespace framework
{

DataProcessorSpec getDCSDataGeneratorSpec()
{
  return DataProcessorSpec{
    "dcs-data-generator",
    Inputs{},
    Outputs{{{"outputDCS"}, "DCS", "DATAPOINTS"}, {{"outputDCSdelta"}, "DCS", "DATAPOINTSdelta"}},
    AlgorithmSpec{adaptFromTask<o2::dcs::DCSDataGenerator>()},
    Options{{"max-timeframes", VariantType::Int64, 99999999999ll, {"max TimeFrames to generate"}}}};
}

} // namespace framework
} // namespace o2

#endif
