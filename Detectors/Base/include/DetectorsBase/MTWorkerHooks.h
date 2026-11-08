// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_BASE_MTWORKERHOOKS_H
#define ALICEO2_BASE_MTWORKERHOOKS_H

#include <functional>
#include <memory>
#include <fairmq/FwdDecls.h>

class TVirtualMCDecayer;

namespace o2::data
{
class PrimaryChunk;
}

namespace o2::base
{

/// What the simulation setup provides to the transport threads of multithreaded Geant4.
/// Installed once on the master thread before transport starts; each function is called on a worker thread.
struct MTWorkerHooks {
  /// Returns the next chunk of primaries for the calling thread, or nullptr when there is no more work
  std::function<std::unique_ptr<o2::data::PrimaryChunk>()> fetchChunk;
  /// Returns the calling thread's channel to the hit merger
  std::function<fair::mq::Channel*()> dataChannel;
  /// Creates the external decayer of the calling thread (empty if there is none)
  std::function<TVirtualMCDecayer*()> makeDecayer;
  /// Ends the event loop of the calling thread after the current event
  std::function<void()> stopWorker;
  /// Called on a worker thread at the start of each run, after its user actions exist
  std::function<void()> beginRunOnWorker;
  /// Called once on the master thread after initialisation and before the first transport
  std::function<void()> beforeTransport;
  /// Called on a worker thread at the start of each track
  std::function<void()> beginTrack;

  static MTWorkerHooks& instance();
};

} // namespace o2::base

#endif
