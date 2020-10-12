// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_SECONDARY_VERTEXER_SPEC_H
#define O2_SECONDARY_VERTEXER_SPEC_H

/// @file SecondaryVertexingSpec.h

#include "DetectorsVertexing/SVertexer.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TStopwatch.h"

namespace o2
{
namespace vertexing
{

using namespace o2::framework;

class SecondaryVertexingSpec : public Task
{
 public:
  SecondaryVertexingSpec() = default;
  ~SecondaryVertexingSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  o2::vertexing::SVertexer mVertexer;
  TStopwatch mTimer;
};

/// create a processor spec
DataProcessorSpec getSecondaryVertexingSpec();

} // namespace vertexing
} // namespace o2

#endif
