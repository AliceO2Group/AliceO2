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

#include <cstdint>
#include <set>
#include <vector>

#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"

namespace o2
{

namespace emcal
{
class TriggerRecord;

/// \class EventBuilderSpec
/// \brief Class combining cell data from different subtimeframes into single cell range
/// \ingroup EMCALEMCALworkflow
/// \author Markus Fasel <markus.fasek@cern.ch>, Oak Ridge National Laboratory
/// \since Aug 9, 2021
///
/// The data processor combines cells from the same trigger in different subtimeframes
/// into a single cell container. Cells within the event range are ordered according
/// to the tower ID. New trigger record objects indicate the combined.
class EventBuilderSpec : public framework::Task
{
 public:
  /// \brief Constructor
  EventBuilderSpec() = default;

  /// \brief Destructor
  ~EventBuilderSpec() override = default;

  void init(framework::InitContext& ctx) final;
  void run(framework::ProcessingContext& ctx) final;

 private:
  struct RangeSubspec {
    header::DataHeader::SubSpecificationType mSpecification;
    dataformats::RangeReference<int, int> mDataRage;
  };

  struct RangeCollection {
    InteractionRecord mInteractionRecord;
    uint32_t mTriggerType;
    std::vector<RangeSubspec> mRangesSubtimeframe;

    bool operator==(const RangeCollection& other) const { return mInteractionRecord == other.mInteractionRecord; }
    bool operator<(const RangeCollection& other) const { return mInteractionRecord < other.mInteractionRecord; }
  };
  std::set<RangeCollection> connectRangesFromSubtimeframes(const std::unordered_map<header::DataHeader::SubSpecificationType, gsl::span<const TriggerRecord>>& triggerrecords) const;
};

o2::framework::DataProcessorSpec getEventBuilderSpec(std::vector<unsigned int> subspecifications);

} // namespace emcal

} // namespace o2
