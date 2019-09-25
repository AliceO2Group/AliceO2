// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MergerCache.cxx
/// \brief Definition of O2 Merger's Cache
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Mergers/MergerCache.h"

namespace o2
{
namespace experimental::mergers
{

//todo: consider storing raw header and payload (not sure if it is deleted by framework)
MergerCache::MergerCache(bool overwrite) : mCache{}, mOverwrite(overwrite)
{
}

void MergerCache::init(const framework::InputRecord& inputs)
{
  // sometimes timer can be one of the inputs, but we do not want to store it in the cache. Therefore,
  // we find its position and omit it when accessing inputs.
  mTimerPosition = inputs.getPos("timer-publish");
  mCache = std::vector<CacheEntryQueue>(inputs.size() - (mTimerPosition >= 0));

  mCachedInputs = 0;
  mUpdatedInputs = 0;
}

void MergerCache::cacheInputRecord(const framework::InputRecord& inputs) //todo: rename to cacheObjects ?
{
  if (uninitialized()) {
    init(inputs);
  }

  bool shift = false;
  for (int i = 0; i < inputs.size(); ++i) {
    // when we iterate besides timer input, we shift the index of cache.
    if (mTimerPosition == i) {
      shift = true;
      continue;
    }

    auto input = inputs.getByPos(i);
    if (input.header && input.payload) {
      auto& queue = mCache[i - shift];

      mUpdatedInputs += !queue.was_updated;
      mCachedInputs += queue.deque.empty();

      queue.was_updated = true;

      std::unique_ptr<TObject, void (*)(TObject*)> objPtr(framework::DataRefUtils::as<TObject>(input).release(), deleteTCollections);

      if (mOverwrite && !queue.deque.empty()) {
        assert(queue.deque.size() == 1);
        queue.deque.front() = {std::move(objPtr)};
      } else {
        queue.deque.push_back({std::move(objPtr)});
      }
    }
  }
}

const MergerCache::CacheEntryQueue& MergerCache::operator[](size_t i) const
{
  return mCache[i];
}

size_t MergerCache::size()
{
  return mCache.size();
}

void MergerCache::clear()
{
  mCachedInputs = 0;
  mUpdatedInputs = 0;
  // We do not use clear() on the whole mCache vector to preserve its size. Instead
  // we clean each queue.
  for (auto& queue : mCache) {
    queue.deque.clear();
    queue.was_updated = false;
  }
}

bool MergerCache::uninitialized()
{
  return mCache.empty();
}

void MergerCache::setMerged(size_t input, size_t entry, bool merged)
{
  mCache[input].deque[entry].is_merged = merged;
}

void MergerCache::setAllMerged(bool merged)
{
  for (auto& queue : mCache) {
    for (auto& entry : queue.deque) {
      entry.is_merged = merged;
    }
  }
}

void MergerCache::setUpdated(size_t i, bool updated)
{
  auto& queue = mCache[i];
  mUpdatedInputs += updated - queue.was_updated;
  queue.was_updated = updated;
}

size_t MergerCache::updatedInputs() { return mUpdatedInputs; }
size_t MergerCache::cachedInputs() { return mCachedInputs; }

void MergerCache::setAllUpdated(bool updated)
{
  for (auto& queue : mCache) {
    mUpdatedInputs += updated - queue.was_updated;
    queue.was_updated = updated;
  }
}

void MergerCache::deleteTCollections(TObject* obj)
{
  // this is not probably the optimal approach, but it should be ok for now
  if (auto c = dynamic_cast<TCollection*>(obj)) {
    c->SetOwner(false);
    auto iter = c->MakeIterator();
    while (auto element = iter->Next()) {
      deleteTCollections(element);
    }
  } else {
    delete obj;
  }
}

} // namespace experimental::mergers
} // namespace o2
