// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MERGERS_H
#define ALICEO2_MERGERS_H

/// \file Merger.h
/// \brief Definition of O2 Merger, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergerConfig.h"
#include "Mergers/MergerCache.h"
#include "Mergers/MergeInterface.h"

#include <Framework/Task.h>

#include <TObject.h>

#include <memory>

namespace o2
{
namespace experimental::mergers
{

/// \brief Merger data processor class.
///
/// Mergers are DPL devices able to merge ROOT objects produced in parallel.
class Merger : public framework::Task
{
 public:
  /// \brief Default constructor. It expects merger configuration and subSpec of output channel.
  Merger(const MergerConfig&, const header::DataHeader::SubSpecificationType&);
  /// \brief Default destructor.
  ~Merger() override = default;

  /// \brief Merger init callback.
  void init(framework::InitContext& ctx) override;
  /// \brief Merger process callback.
  void run(framework::ProcessingContext& ctx) override;

 private:
  // todo: maybe the pointer vector to the extracted objects should be stored all the time
  std::function<void()> prepareTimerCallback(framework::InitContext& ictx) const;
  std::vector<TObject*> unpackObjects(TObject* obj);
  void mergeCache();
  void publish(framework::DataAllocator& allocator);

  void cleanCacheAfterMerging();
  void cleanCacheAfterPublishing();

  bool shouldPublish(framework::ProcessingContext&);
  bool shouldMergeCache(framework::ProcessingContext& ctx);

 private:
  header::DataHeader::SubSpecificationType mSubSpec;
  MergerCache mCache;
  std::unique_ptr<TObject> mMergedObjects;
  MergerConfig mConfig;
};

} // namespace experimental::mergers
} // namespace o2

#endif //ALICEO2_MERGERS_H
