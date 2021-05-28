// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitRecoSpec.cxx
/// @brief  ZDC reconstruction
/// @author pietro.cortese@cern.ch

#include <vector>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "ZDCWorkflow/DigitRecoSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "FairLogger.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

DigitRecoSpec::DigitRecoSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

void DigitRecoSpec::init(o2::framework::InitContext& ic)
{
  // At this stage we cannot access the CCDB yet
  //   std::string dictPath = ic.options().get<std::string>("zdc-ctf-dictionary");
  //   if (!dictPath.empty() && dictPath != "none") {
  //     mCTFCoder.createCoders(dictPath, o2::ctf::CTFCoderBase::OpType::Encoder);
  //   }
}

void DigitRecoSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mInitialized = true;
    // Initialization from CCDB
  }
  auto cput = mTimer.CpuTime();
  mTimer.Start(false);
  auto bcdata = pc.inputs().get<gsl::span<o2::zdc::BCData>>("trig");
  auto chans = pc.inputs().get<gsl::span<o2::zdc::ChannelData>>("chan");
  auto peds = pc.inputs().get<gsl::span<o2::zdc::OrbitData>>("peds");

  //   auto& buffer = pc.outputs().make<std::vector<o2::ctf::BufferType>>(Output{"ZDC", "CTFDATA", 0, Lifetime::Timeframe});
  //   mCTFCoder.encode(buffer, bcdata, chans, peds);
  //   auto eeb = CTF::get(buffer.data()); // cast to container pointer
  //   eeb->compactify();                  // eliminate unnecessary padding
  //   buffer.resize(eeb->size());         // shrink buffer to strictly necessary size
  //  eeb->print();
  mTimer.Stop();
  //  LOG(INFO) << "Created encoded data of size " << eeb->size() << " for ZDC in " << mTimer.CpuTime() - cput << " s";
}

void DigitRecoSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(INFO, "ZDC Reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getDigitRecoSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trig", "ZDC", "DIGITSBC", 0, Lifetime::Timeframe);
  inputs.emplace_back("chan", "ZDC", "DIGITSCH", 0, Lifetime::Timeframe);
  inputs.emplace_back("peds", "ZDC", "DIGITSPD", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "zdc-digi-reco",
    inputs,
    Outputs{{"ZDC", "RECODATA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<DigitRecoSpec>()}};
}

} // namespace zdc
} // namespace o2
