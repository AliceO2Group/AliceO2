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

#ifndef O2_ITS_TESTDATAGETTER
#define O2_ITS_TESTDATAGETTER

#include <fstream>

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

class TestDataGetter : public Task
{
 public:
  TestDataGetter() = default;
  ~TestDataGetter() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
};

/// create a processor spec
/// run ITS cluster finder
framework::DataProcessorSpec getTestDataGetterSpec();

} // namespace its
} // namespace o2

#endif
