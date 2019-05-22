// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_ANALYSIS_TASK_H_
#define FRAMEWORK_ANALYSIS_TASK_H_

#include "Framework/DataProcessorSpec.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/Traits.h"
#include "Framework/Kernels.h"
#include "Framework/ASoA.h"

#include <arrow/compute/context.h>
#include <arrow/compute/kernel.h>
#include <arrow/table.h>
#include <type_traits>
#include <utility>
#include <memory>

namespace o2
{

namespace aod
{

namespace track
{
DECLARE_SOA_COLUMN(CollisionId, collisionId, int, "fID4Tracks");
DECLARE_SOA_COLUMN(X, x, float, "fX");
DECLARE_SOA_COLUMN(Alpha, alpha, float, "fAlpha");
DECLARE_SOA_COLUMN(Y, y, float, "fY");
DECLARE_SOA_COLUMN(Z, z, float, "fZ");
DECLARE_SOA_COLUMN(Snp, snp, float, "fSnp");
DECLARE_SOA_COLUMN(Tgl, tgl, float, "fTgl");
DECLARE_SOA_COLUMN(Signed1Pt, signed1Pt, float, "fSigned1Pt");
} // namespace track

using Tracks = soa::Table<track::CollisionId, track::X, track::Alpha,
                          track::Y, track::Z, track::Snp, track::Tgl,
                          track::Signed1Pt>;
using Track = Tracks::iterator;

namespace collision
{
DECLARE_SOA_COLUMN(Centrality, centrality, float, "centrality");
} // namespace collision

using Collisions = soa::Table<collision::Centrality>;
using Collision = Collisions::iterator;

namespace timeframe
{
DECLARE_SOA_COLUMN(Timestamp, timestamp, uint64_t, "timestamp");
} // namespace timeframe

using Timeframes = soa::Table<timeframe::Timestamp>;
using Timeframe = Timeframes::iterator;

} // namespace aod

namespace framework
{

/// A more familiar task API for the DPL analysis framework.
/// This allows you to define your own tasks as subclasses
/// of o2::framework::AnalysisTask and to pass them in the specification
/// using:
///
/// adaptAnalysisTask<YourDerivedTask>(constructor args, ...);
///
/// The appropriate AlgorithmSpec invoking `AnalysisTask::init(...)` at
/// startup and `AnalysisTask::run(...)` will be created.
///
class AnalysisTask
{
 public:
  virtual ~AnalysisTask() = default;
  /// The method which is called once to initialise the task.
  /// Derived classes can use this to save extra state.
  virtual void init(InitContext& context) {}
  /// This is invoked whenever a new InputRecord is demeed to
  /// be complete.
  virtual void run(ProcessingContext& context) = 0;

  /// Override this to subscribe to each track. No guarantees on the order.
  virtual void processTrack(aod::Track const& tracks) {}
  /// Override this to subscribe to all the tracks and including their associated collision.
  virtual void processCollisionTrack(aod::Collision const& collision, aod::Track const& track) {}
  /// Override this to subscribe to all the tracks of the given timeframe.
  virtual void processTimeframeTracks(aod::Timeframe const& timeframe, aod::Tracks const& tracks) {}
  /// Override this to subscribe to all the tracks associated to a given collision.
  virtual void processCollisionTracks(aod::Collision const& collision, aod::Tracks const& tracks) {}
};

/// Adaptor to make an AlgorithmSpec from a o2::framework::Task
///
template <typename T, typename... Args>
DataProcessorSpec adaptAnalysisTask(std::string name, Args&&... args)
{
  constexpr bool hasProcessTrack = is_overriding<decltype(&T::processTrack),
                                                 decltype(&AnalysisTask::processTrack)>::value;
  constexpr bool hasProcessCollisionTrack = is_overriding<decltype(&T::processCollisionTrack),
                                                          decltype(&AnalysisTask::processCollisionTrack)>::value;
  constexpr bool hasProcessTimeframeTracks = is_overriding<decltype(&T::processTimeframeTracks),
                                                           decltype(&AnalysisTask::processTimeframeTracks)>::value;
  constexpr bool hasProcessCollisionTracks = is_overriding<decltype(&T::processCollisionTracks),
                                                           decltype(&AnalysisTask::processCollisionTracks)>::value;

  auto task = std::make_shared<T>(std::forward<Args>(args)...);
  auto algo = AlgorithmSpec::InitCallback{ [task](InitContext& ic) {
    task->init(ic);
    return [task](ProcessingContext& pc) {
      task->run(pc);
      if constexpr (hasProcessTrack) {
        auto tracks = pc.inputs().get<TableConsumer>("tracks");
        for (auto& track : aod::Tracks(tracks->asArrowTable())) {
          task->processTrack(track);
        }
      }
      if constexpr (hasProcessCollisionTrack) {
        auto tracks = pc.inputs().get<TableConsumer>("tracks");
        auto collisions = pc.inputs().get<TableConsumer>("collisions");
        size_t currentCollision = 0;
        aod::Collision collision(collisions->asArrowTable());
        for (auto& track : aod::Tracks(tracks->asArrowTable())) {
          auto collisionIndex = track.collisionId();
          // We find the associated collision, assuming they are sorted.
          while (collisionIndex > currentCollision) {
            ++currentCollision;
            ++collision;
          }
          task->processTrack(collision, track);
        }
      }
      if constexpr (hasProcessTimeframeTracks) {
        auto tracks = pc.inputs().get<TableConsumer>("tracks");
        auto timeframes = pc.inputs().get<TableConsumer>("timeframe");
        // FIXME: For the moment we assume we have a single timeframe...
        aod::Timeframe timeframe(timeframes->asArrowTable());
        task->processTimeframeTracks(timeframe, aod::Tracks(tracks->asArrowTable()));
      }
      if constexpr (hasProcessCollisionTracks) {
        auto collisions = pc.inputs().get<TableConsumer>("collisions");
        auto allTracks = pc.inputs().get<TableConsumer>("tracks");
        arrow::compute::FunctionContext ctx;
        std::vector<arrow::compute::Datum> eventTracksCollection;
        auto result = o2::framework::sliceByColumn(&ctx, "fID4Tracks", allTracks->asArrowTable(), &eventTracksCollection);
        if (result.ok() == false) {
          LOG(ERROR) << "Error while splitting the tracks per events";
          return;
        }
        size_t currentCollision = 0;
        aod::Collision collision(collisions->asArrowTable());
        for (auto& eventTracks : eventTracksCollection) {
          // FIXME: We find the associated collision, assuming they are sorted.
          aod::Tracks tracks(arrow::util::get<std::shared_ptr<arrow::Table>>(eventTracks.value));
          auto collisionIndex = tracks.begin().collisionId();
          while (collisionIndex > currentCollision) {
            ++currentCollision;
            ++collision;
          }
          task->processCollisionTracks(collision, tracks);
        }
      }
    };
  } };
  std::vector<InputSpec> inputs;

  if constexpr (hasProcessTrack || hasProcessCollisionTrack || hasProcessTimeframeTracks || hasProcessCollisionTracks) {
    inputs.emplace_back(InputSpec{ "tracks", "AOD", "TRACKPAR" });
  }
  if constexpr (hasProcessCollisionTrack || hasProcessCollisionTracks) {
    inputs.emplace_back(InputSpec{ "collisions", "AOD", "COLLISIONS" });
  }
  if constexpr (hasProcessTimeframeTracks) {
    inputs.emplace_back(InputSpec{ "timeframe", "AOD", "TIMEFRAME" });
  }

  DataProcessorSpec spec{
    name,
    // FIXME: For the moment we hardcode this. We could build
    // this list from the list of methods actually implemented in the
    // task itself.
    inputs,
    // FIXME: Placeholeder for results. We should make it configurable
    // from the task.
    Outputs{ OutputSpec{ "ASIS", "RESULTS", 0 } },
    algo
  };
  return spec;
}

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_ANALYSISTASK_H_
