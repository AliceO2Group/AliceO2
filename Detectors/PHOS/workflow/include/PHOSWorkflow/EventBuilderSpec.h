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

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{

namespace phos
{
class TriggerRecord;
class Cell;

/// \class EventBuilderSpec
/// \brief Class merges subevents from two FLPs
/// \ingroup PHOSworkflow
/// \author Dmitri Peresunko, NRC "Kurchatov institute"
/// \since March, 2022
///
/// Merge subevents from two FLPs. FLPs send messages with non-zero subspecifications
/// take TrigRecs with same BC stamps and copy corresponding cells
/// sells are sorted, so find subspec with smaller absId and copy it first (exclude/handle trigger cells!)
/// check if all halves of events were found, otherwise send warning
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
  class SubspecSet
  {
   public:
    SubspecSet(gsl::span<const o2::phos::TriggerRecord> r, gsl::span<const o2::phos::Cell> c)
    {
      trSpan = r;
      cellSpan = c;
    }
    ~SubspecSet() = default;
    gsl::span<const o2::phos::TriggerRecord> trSpan;
    gsl::span<const o2::phos::Cell> cellSpan;
  };
};

o2::framework::DataProcessorSpec getEventBuilderSpec();

} // namespace phos

} // namespace o2
