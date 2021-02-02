// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_PRIMARY_VERTEXER_SPEC_H
#define O2_PRIMARY_VERTEXER_SPEC_H

/// @file PrimaryVertexingSpec.h

#include "DetectorsVertexing/PVertexer.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "TStopwatch.h"

namespace o2
{
namespace vertexing
{

using namespace o2::framework;

class PrimaryVertexingSpec : public Task
{
 public:
  PrimaryVertexingSpec(bool validateWithIR, bool useMC) : mUseMC(useMC), mValidateWithIR(validateWithIR) {}
  ~PrimaryVertexingSpec() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(EndOfStreamContext& ec) final;

 private:
  o2::vertexing::PVertexer mVertexer;
  bool mUseMC{false};           ///< MC flag
  bool mValidateWithIR{false};  ///< require vertex validation with IR (e.g. from FT0)
  TStopwatch mTimer;
};

/// create a processor spec
DataProcessorSpec getPrimaryVertexingSpec(bool validateWithFT0, bool useMC);

} // namespace vertexing
} // namespace o2

#endif
