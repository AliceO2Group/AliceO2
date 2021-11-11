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

/// @file   StandaloneAODProducerSpec.h
/// @brief  Convert CTF (EncodedBlocks) to AO2D EMCal standalone

#ifndef O2_EMCAL_STANDALONEAODPRODUCERSPEC_SPEC
#define O2_EMCAL_STANDALONEAODPRODUCERSPEC_SPEC

#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisHelpers.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsEMCAL/EventHandler.h"
#include <TStopwatch.h>

namespace o2
{
namespace emcal
{

class StandaloneAODProducerSpec : public o2::framework::Task
{
 public:
  StandaloneAODProducerSpec();
  ~StandaloneAODProducerSpec() override = default;
  void run(o2::framework::ProcessingContext& pc) final;
  void init(o2::framework::InitContext& ic) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;

  static const char* getCellBinding() { return "EMCCells"; }
  static const char* getCellTriggerRecordBinding() { return "EMCCellsTrgR"; }

 private:
  int64_t mTFNumber = -1;                                                // Timeframe ID
  int mRunNumber = -1;                                                   // Run number
  uint32_t mCaloAmp = 0xFFFFFF00;                                        // 15 bits
  uint32_t mCaloTime = 0xFFFFFF00;                                       // 15 bits
  o2::emcal::EventHandler<o2::emcal::Cell>* mCaloEventHandler = nullptr; ///< Pointer to the event builder for emcal cells
  TStopwatch mTimer;

  static const char* BININGCELLS;
  static const char* BINDINGCELLSTRG;
};

/// create a processor spec
framework::DataProcessorSpec getStandaloneAODProducerSpec();

} // namespace emcal
} // namespace o2

#endif