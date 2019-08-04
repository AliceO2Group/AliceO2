// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_MERGERCACHE_H
#define ALICEO2_MERGERCACHE_H

/// \file MergerCache.h
/// \brief Definition of O2 Merger cache, v0.1
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <Framework/InputRecord.h>

#include <TObject.h>

#include <memory>
#include <vector>
#include <deque>

namespace o2
{
namespace experimental::mergers
{

/// \brief Merger cache to store input objects before merging them.
class MergerCache
{
  /// \brief Cache entry storing one mergable object.
  struct CacheEntry {
    std::unique_ptr<TObject, void (*)(TObject*)> obj;
    bool is_merged = false;
  };

  /// \biref Cache entry queue storing multiple mergable objects received at one input.
  struct CacheEntryQueue {
    std::deque<CacheEntry> deque;
    bool was_updated = false;
  };

 public:
  /// \brief Default constructor. When overwrite == true, only one object per input is stored.
  MergerCache(bool overwrite = true);
  /// \brief Default destructor.
  ~MergerCache() = default;

  /// \brief Stores input objects from InputRecord into the cache
  void cacheInputRecord(const framework::InputRecord& inputs);

  size_t size();
  void clear();
  bool uninitialized();
  const CacheEntryQueue& operator[](size_t i) const;

  using const_iterator = std::vector<CacheEntryQueue>::const_iterator;
  const_iterator begin() const { return mCache.begin(); }
  const_iterator end() const { return mCache.end(); }

  void setMerged(size_t i, size_t entry, bool merged = true);
  void setAllMerged(bool merged = true);
  void setUpdated(size_t i, bool updated = true);
  void setAllUpdated(bool updated = true);

  // returns the number of inputs that have been updated since the last time the flag was cleared (last publication)
  size_t updatedInputs();
  // returns the number of inputs that have some corresponding objects cached
  size_t cachedInputs();

 private:
  /// \brief Initializes the cache by seeing the size and contents of DPL's InputRecord
  void init(const framework::InputRecord& inputs);

  static void deleteTCollections(TObject* obj);

  std::vector<CacheEntryQueue> mCache;
  // when active, cache keeps only one entry per queue (per input)
  bool mOverwrite;
  // keeps the index of the timer in the InputRecord
  int mTimerPosition{-1};
  // counts how many inputs have some corresponding objects cached
  size_t mCachedInputs{0};
  // counts how many inputs have been updated since the last time the flag was cleared (last publication)
  size_t mUpdatedInputs{0};
};

} // namespace experimental::mergers
} // namespace o2

#endif //ALICEO2_MERGERCACHE_H
