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

#ifndef O2_MCKINEMATIC_READER_SPEC_H
#define O2_MCKINEMATIC_READER_SPEC_H

#include <Steer/MCKinematicsReader.h>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace steer
{
using namespace o2::framework;
class KinematicReader : public Task
{
 public:
  KinematicReader() : mMCKinReader(std::make_unique<MCKinematicsReader>()), mCurrentEvent(0), mNEvents(0){};
  ~KinematicReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 protected:
  std::unique_ptr<MCKinematicsReader> mMCKinReader;
  std::string mKineFileName = "o2sim";
  int mCurrentEvent;
  int mNEvents;
};

DataProcessorSpec getMCKinematicReaderSpec();
} // namespace steer
} // namespace o2
#endif