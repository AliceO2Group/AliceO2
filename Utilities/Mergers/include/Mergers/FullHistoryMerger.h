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

#ifndef ALICEO2_FULLHISTORYMERGER_H
#define ALICEO2_FULLHISTORYMERGER_H

/// \file FullHistoryMerger.h
/// \brief Definition of O2 FullHistoryMerger, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergerConfig.h"
#include "Mergers/ObjectStore.h"

#include <Framework/Task.h>

namespace o2::monitoring
{
class Monitoring;
}

namespace o2::mergers
{

/// \brief FullHistoryMerger data processor class.
///
/// Mergers are DPL devices able to merge objects produced in parallel.
class FullHistoryMerger : public framework::Task
{
 public:
  /// \brief Default constructor. It expects Merger configuration and subSpec of output channel.
  FullHistoryMerger(const MergerConfig&, const header::DataHeader::SubSpecificationType&);
  /// \brief Default destructor.
  ~FullHistoryMerger() override;

  /// \brief FullHistoryMerger init callback.
  void init(framework::InitContext& ctx) override;
  /// \brief FullHistoryMerger process callback.
  void run(framework::ProcessingContext& ctx) override;

  /// \brief Callback for CallbackService::Id::EndOfStream
  void endOfStream(framework::EndOfStreamContext& eosContext) override;

 private:
  header::DataHeader::SubSpecificationType mSubSpec;

  ObjectStore mMergedObject = std::monostate{};
  std::pair<std::string, framework::DataRef> mFirstObjectSerialized;
  std::unordered_map<std::string, ObjectStore> mCache;

  MergerConfig mConfig;
  std::unique_ptr<monitoring::Monitoring> mCollector;
  int mCyclesSinceReset = 0;

  // stats
  int mTotalObjectsMerged = 0;
  int mObjectsMerged = 0;
  int mTotalUpdatesReceived = 0;
  int mUpdatesReceived = 0;

 private:
  void updateCache(const framework::DataRef& ref);
  void mergeCache();
  void publish(framework::DataAllocator& allocator);
  void clear();
};

} // namespace o2::mergers

#endif //ALICEO2_FULLHISTORYMERGER_H
