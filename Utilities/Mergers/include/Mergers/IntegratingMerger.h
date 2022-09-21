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

#ifndef ALICEO2_INTEGRATINGMERGER_H
#define ALICEO2_INTEGRATINGMERGER_H

/// \file IntegratingMerger.h
/// \brief Definition of O2 IntegratingMerger, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergerConfig.h"
#include "Mergers/MergeInterface.h"
#include "Mergers/ObjectStore.h"

#include "Framework/Task.h"

#include <memory>

class TObject;

namespace o2::monitoring
{
class Monitoring;
}

namespace o2::mergers
{

/// \brief IntegratingMerger data processor class.
///
/// Mergers are DPL devices able to merge ROOT objects produced in parallel.
class IntegratingMerger : public framework::Task
{
 public:
  /// \brief Default constructor. It expects Merger configuration and subSpec of output channel.
  IntegratingMerger(const MergerConfig&, const header::DataHeader::SubSpecificationType&);
  /// \brief Default destructor.
  ~IntegratingMerger() override = default;

  /// \brief IntegratingMerger init callback.
  void init(framework::InitContext& ctx) override;
  /// \brief IntegratingMerger process callback.
  void run(framework::ProcessingContext& ctx) override;

  /// \brief Callback for CallbackService::Id::EndOfStream
  void endOfStream(framework::EndOfStreamContext& eosContext) override;

 private:
  void publish(framework::DataAllocator& allocator);
  void clear();

 private:
  header::DataHeader::SubSpecificationType mSubSpec;
  ObjectStore mMergedObject = std::monostate{};
  MergerConfig mConfig;
  std::unique_ptr<monitoring::Monitoring> mCollector;
  int mCyclesSinceReset = 0;

  // stats
  int mTotalDeltasMerged = 0;
  int mDeltasMerged = 0;
};

} // namespace o2::mergers

#endif //ALICEO2_INTEGRATINGMERGER_H
