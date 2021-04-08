// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   centralEventFilterProcessor.h

#ifndef O2_CentralEventFilterProcessor
#define O2_CentralEventFilterProcessor

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "filterTables.h"

namespace o2::aod::filtering
{

class CentralEventFilterProcessor : public framework::Task
{
 public:
  CentralEventFilterProcessor(const std::string &config) : mConfigFile{config} {}
  ~CentralEventFilterProcessor() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final {}

 private:
  std::string mConfigFile;
  std::unordered_map<std::string, float> mDownscaling;
};

/// create a processor spec
framework::DataProcessorSpec getCentralEventFilterProcessorSpec(std::string &config);

} // namespace o2::aod::filtering

#endif /* O2_CentralEventFilterProcessor */
