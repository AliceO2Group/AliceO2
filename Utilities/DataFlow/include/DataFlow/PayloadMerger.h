// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef PAYLOAD_MERGER_H
#define PAYLOAD_MERGER_H

#include <map>
#include <cstdint>
#include <vector>
#include <functional>
#include <cstring>

#include <fairmq/FairMQMessage.h>

namespace o2
{
namespace dataflow
{
/// Helper class that given a set of FairMQMessage, merges (part of) their
/// payload into a separate memory area.
///
/// - Append multiple messages via the aggregate method
/// - Finalise buffer creation with the finalise call.
template <typename ID>
class PayloadMerger
{
 public:
  using MergeableId = ID;
  using MessageMap = std::multimap<MergeableId, std::unique_ptr<FairMQMessage>>;
  using PayloadExtractor = std::function<size_t(char**, char*, size_t)>;
  using IdExtractor = std::function<MergeableId(std::unique_ptr<FairMQMessage>&)>;
  using MergeCompletionCheker = std::function<bool(MergeableId, MessageMap&)>;

  /// Helper class to merge FairMQMessages sharing a user defined class of equivalence,
  /// specified by @makeId. Completeness of the class of equivalence can be asserted by
  /// the @checkIfComplete policy. It's also possible to specify a user defined way of
  /// extracting the parts of the payload to be merged via the extractPayload method.
  PayloadMerger(IdExtractor makeId,
                MergeCompletionCheker checkIfComplete,
                PayloadExtractor extractPayload = fullPayloadExtractor)
    : mMakeId{makeId},
      mCheckIfComplete{checkIfComplete},
      mExtractPayload{extractPayload}
  {
  }

  /// Aggregates @payload to all the ones with the same id.
  /// @return the id extracted from the payload via the constructor
  ///         specified id policy (mMakeId callback).
  MergeableId aggregate(std::unique_ptr<FairMQMessage>& payload)
  {
    auto id = mMakeId(payload);
    mPartsMap.emplace(std::make_pair(id, std::move(payload)));
    return id;
  }

  /// This merges a set of messages sharing the same id @id to a unique buffer
  /// @out, so that it can be either consumed or sent as a message itself.
  /// The decision on whether the merge must happen is done by the constructor
  /// specified policy mCheckIfComplete which can, for example, decide
  /// to merge when a certain number of subparts are reached.
  /// Merging at the moment requires an extra copy, but in principle this could
  /// be easily extended to support scatter - gather.
  size_t finalise(char** out, MergeableId& id)
  {
    *out = nullptr;
    if (mCheckIfComplete(id, mPartsMap) == false) {
      return 0;
    }
    // If we are here, it means we can send the messages that belong
    // to some predefined class of equivalence, identified by the MERGEABLE_ID,
    // to the receiver. This is done by the following process:
    //
    // - Extract what we actually want to send (this might be data embedded inside the message itself)
    // - Calculate the aggregate size of all the payloads.
    // - Copy all the parts into a final payload
    // - Create the header part
    // - Create the payload part
    // - Send
    std::vector<std::pair<char*, size_t>> parts;

    size_t sum = 0;
    auto range = mPartsMap.equal_range(id);
    for (auto hi = range.first, he = range.second; hi != he; ++hi) {
      std::unique_ptr<FairMQMessage>& payload = hi->second;
      std::pair<char*, size_t> part;
      part.second = mExtractPayload(&part.first, reinterpret_cast<char*>(payload->GetData()), payload->GetSize());
      parts.push_back(part);
      sum += part.second;
    }

    auto* payload = new char[sum]();
    size_t offset = 0;
    for (auto& part : parts) {
      // Right now this does a copy. In principle this could be done with some sort of
      // vectorized I/O
      memcpy(payload + offset, part.first, part.second);
      offset += part.second;
    }

    mPartsMap.erase(id);
    *out = payload;
    return sum;
  }

  // Helper method which leaves the payload untouched
  static int64_t fullPayloadExtractor(char** payload,
                                      char* buffer,
                                      size_t bufferSize)
  {
    *payload = buffer;
    return bufferSize;
  }

 private:
  IdExtractor mMakeId;
  MergeCompletionCheker mCheckIfComplete;
  PayloadExtractor mExtractPayload;

  MessageMap mPartsMap;
};
} // namespace dataflow
} // namespace o2

#endif // PAYLOAD_MERGER_H
