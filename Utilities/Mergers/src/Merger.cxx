// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Merger.cxx
/// \brief Implementation of O2 Mergers, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/Merger.h"
#include "Mergers/MergerBuilder.h"

#include <Framework/CompletionPolicyHelpers.h>
#include <Framework/TimesliceIndex.h>
#include <Framework/CallbackService.h>

#include <TObjArray.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THn.h>
#include <TTree.h>
#include <THnSparse.h>

using namespace o2::framework;
using namespace std::chrono;

namespace o2
{
namespace experimental::mergers
{

Merger::Merger(MergerConfig config, header::DataHeader::SubSpecificationType subSpec)
  : mConfig(config),
    mSubSpec(subSpec),
    mCache(config.ownershipMode.value == OwnershipMode::Full)
{
}

void Merger::init(framework::InitContext& ictx)
{
  if (mConfig.publicationDecision.value == PublicationDecision::EachNSeconds) {
    // Register a device callback which creates timeslice in the TimesliceIndex
    // each N seconds, so it can serve as timer input.
    ictx.services().get<CallbackService>().set(CallbackService::Id::ClockTick, prepareTimerCallback(ictx));
  }
}

void Merger::run(framework::ProcessingContext& ctx)
{
  mCache.cacheInputRecord(ctx.inputs());

  if (shouldMergeCache(ctx)) {

    mergeCache();

    cleanCacheAfterMerging();
  }

  if (shouldPublish(ctx)) {

    publish(ctx.outputs());

    cleanCacheAfterPublishing();

    if (mConfig.timespan.value == Timespan::LastDifference) {
      mMergedObjects.reset();
    }
  }
}

std::function<void()> Merger::prepareTimerCallback(InitContext& ictx) const
{
  return [&timesliceIndex = ictx.services().get<TimesliceIndex>(),
          timeLast = std::make_shared<steady_clock::time_point>(steady_clock::now()),
          periodMs = this->mConfig.publicationDecision.param * 1000,
          timesliceID = uint64_t(0)]() mutable {
    auto timeNow = steady_clock::now();

    if (duration_cast<milliseconds>(timeNow - *timeLast).count() > periodMs) {

      data_matcher::VariableContext context;
      context.put(data_matcher::ContextUpdate{0, timesliceID});
      context.commit();

      timesliceIndex.replaceLRUWith(context);

      timesliceID++;
      *timeLast = timeNow;
    }
  };
}

void Merger::cleanCacheAfterMerging()
{
  mCache.setAllMerged();
}

void Merger::cleanCacheAfterPublishing()
{
  if (mConfig.timespan.value == Timespan::LastDifference || mConfig.ownershipMode.value == OwnershipMode::Integral) {
    mCache.clear();
  }

  if (mConfig.ownershipMode.value == OwnershipMode::Full) {
    mCache.setAllMerged(false);
  }
  mCache.setAllUpdated(false);
}

void Merger::mergeCache()
{
  switch (mConfig.mergingMode.value) {
    case MergingMode::Binwise: {

      size_t i = 0;
      if (!mMergedObjects) {
        for (; i < mCache.size(); i++) {
          if (!mCache[i].deque.empty()) {
            mMergedObjects.reset(mCache[i].deque[0].obj->Clone());
            mCache.setMerged(i, 0);
            break;
          }
        }
      }

      if (!mMergedObjects) {
        LOG(INFO) << "mergeCache(): The cache is empty, nothing to merge.";
        return;
      }

      auto unpackedMergedObjects = unpackObjects(mMergedObjects.get());
      std::vector<TObjArray> unpackedCollectionsOfObjects(unpackedMergedObjects.size());
      // todo: unpack straight to TCollection?
      for (; i < mCache.size(); i++) {
        for (const auto& entry : mCache[i].deque) {
          if (!entry.is_merged) {
            auto unpackedCachedObjects = unpackObjects(entry.obj.get());
            assert(unpackedMergedObjects.size() == unpackedCachedObjects.size());

            for (int j = 0; j < unpackedCachedObjects.size(); j++) {
              unpackedCollectionsOfObjects[j].Add(unpackedCachedObjects[j]);
            }
          }
        }
      }

      for (int k = 0; k < unpackedMergedObjects.size(); k++) {

        TObject* mergedObject = unpackedMergedObjects[k];
        const char* className = mergedObject->ClassName();
        Long64_t errorCode = 0;

        //todo: investigate -NOCHECK flag for histogram merging
        auto objectMergeInterface = dynamic_cast<MergeInterface*>(mergedObject);
        if (objectMergeInterface) {
          errorCode = objectMergeInterface->merge(&unpackedCollectionsOfObjects[k]);
        } else if (strncmp(className, "TH1", 3) == 0) {
          errorCode = reinterpret_cast<TH1*>(mergedObject)->Merge(&unpackedCollectionsOfObjects[k]);
        } else if (strncmp(className, "TH2", 3) == 0) {
          errorCode = reinterpret_cast<TH2*>(mergedObject)->Merge(&unpackedCollectionsOfObjects[k]);
        } else if (strncmp(className, "TH3", 3) == 0) {
          errorCode = reinterpret_cast<TH3*>(mergedObject)->Merge(&unpackedCollectionsOfObjects[k]);
        } else if (strncmp(className, "THn", 3) == 0) {
          errorCode = reinterpret_cast<THn*>(mergedObject)->Merge(&unpackedCollectionsOfObjects[k]);
        } else if (strncmp(className, "THnSparse", 8) == 0) {
          errorCode = reinterpret_cast<THnSparse*>(mergedObject)->Merge(&unpackedCollectionsOfObjects[k]);
        } else if (strcmp(className, "TTree") == 0) {
          errorCode = reinterpret_cast<TTree*>(mergedObject)->Merge(&unpackedCollectionsOfObjects[k]);
        } else {
          //          LOG(ERROR) << "Object with type " << className << " is not one of mergeable type.";
          throw std::runtime_error("Object with type '" + std::string(className) + "' is not one of mergeable type.");
          // todo: maybe it is fine to just overwrite?
        }

        if (errorCode == -1) {
          throw std::runtime_error("Binwise merging object of type '" + std::string(className) + "' failed.");
          //          LOG(ERROR) << "Merging object of type " << className << " failed";
          return;
        }
      }

      break;
    }
    case MergingMode::Concatenate: {

      if (!mMergedObjects) {
        mMergedObjects = std::make_unique<TObjArray>();
      }
      for (const auto& queue : mCache) {
        for (const auto& entry : queue.deque) {
          if (!entry.is_merged) {

            //todo: optimisations - check only once if this is a collection
            //todo: optimisations - TCollection::Merge() falls back to slow TMethodCall! (from mikolaj's presentation)
            TCollection* entryAsCollection = dynamic_cast<TCollection*>(entry.obj.get());
            assert(mMergedObjects);
            if (entryAsCollection) {
              reinterpret_cast<TCollection*>(mMergedObjects.get())->AddAll(entryAsCollection);
            } else {
              reinterpret_cast<TCollection*>(mMergedObjects.get())->Add(entry.obj.get());
            }
          }
        }
      }

      break;
    }
    case MergingMode::Timewise:
      throw std::runtime_error("MergingMode::Timewise not supported yet");
    default:
      break;
  }
}

void Merger::publish(framework::DataAllocator& allocator)
{
  if (mMergedObjects) {
    if (mConfig.ownershipMode.value == OwnershipMode::Integral) {
      allocator.snapshot(framework::OutputRef{MergerBuilder::mergerOutputBinding(), mSubSpec}, *mMergedObjects.get());
    } else if (mConfig.ownershipMode.value == OwnershipMode::Full) {
      allocator.adopt(framework::OutputRef{MergerBuilder::mergerOutputBinding(), mSubSpec}, mMergedObjects.release());
    }
  }
}

std::vector<TObject*> Merger::unpackObjects(TObject* obj)
{
  if (auto objMergeInterface = dynamic_cast<MergeInterface*>(obj)) {
    return objMergeInterface->unpack();
  } else if (mConfig.unpackingMethod.value == UnpackingMethod::NoUnpackingNeeded) {
    return std::vector<TObject*>{obj};
  } else if (mConfig.unpackingMethod.value == UnpackingMethod::TCollection) {
    // todo: this could be also checked by casting
    return {};
  } else {
    return {};
  }
}

bool Merger::shouldMergeCache(framework::ProcessingContext& ctx)
{
  switch (mConfig.mergingTime.value) {
    case MergingTime::AfterArrival:
      return true;
    case MergingTime::WhenXInputsCached:
      return double(mCache.cachedInputs()) / mCache.size() >= mConfig.mergingTime.param;
    case MergingTime::BeforePublication:
      return shouldPublish(ctx);
    default:
      return false;
  }
}

bool Merger::shouldPublish(framework::ProcessingContext& ctx)
{
  switch (mConfig.publicationDecision.value) {
    case PublicationDecision::EachNSeconds:
      return ctx.inputs().isValid("timer-publish");
    case PublicationDecision::WhenXInputsUpdated:
      return double(mCache.updatedInputs()) / mCache.size() >= mConfig.publicationDecision.param;
    default:
      return false;
  }
}

} // namespace experimental::mergers
} // namespace o2
