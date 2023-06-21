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

#include <ITSStudies/MCKinematicReaderSpec.h>
#include <Framework/ControlService.h>

namespace o2
{
namespace steer
{
using namespace o2::framework;

void KinematicReader::init(o2::framework::InitContext& ic)
{
  mMCKinReader->initFromKinematics(std::string(mKineFileName));
  mNEvents = mMCKinReader->getNEvents(0);
  LOGP(info, "Initialised KinematicReader");
}

void KinematicReader::run(ProcessingContext& pc)
{
  std::vector<o2::dataformats::MCEventHeader> tfMCHeaders;
  std::vector<o2::MCTrack> tfMCTracks;
  for (int iEvent{0}; iEvent < mNEvents; iEvent++) { // Single TF loaded from File, might extend it to use source
    auto mcHeader = mMCKinReader->getMCEventHeader(0, iEvent);
    auto mcTracks = mMCKinReader->getTracks(0, iEvent);
    tfMCHeaders.push_back(mcHeader);
    tfMCTracks.insert(tfMCTracks.end(), mcTracks.begin(), mcTracks.end());
  }
  // pc.outputs().snapshot(Output{"MC", "MCHEADER", 0, Lifetime::Timeframe}, tfMCHeaders); TODO: not messageable, what can we do?
  pc.outputs().snapshot(Output{"MC", "MCTRACKS", 0, Lifetime::Timeframe}, tfMCTracks);
  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getMCKinematicReaderSpec()
{
  std::vector<OutputSpec> outputSpec;
  // outputSpec.emplace_back("MC", "MCHEADER", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("MC", "MCTRACKS", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "mc-kinematic-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<KinematicReader>()},
    Options{
      {"kineFileName", VariantType::String, "o2sim", {"Name of the input Kine file"}}}};
}
} // namespace steer
} // namespace o2