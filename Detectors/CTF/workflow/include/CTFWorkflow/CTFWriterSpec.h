// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CTFWriterSpec.h

#ifndef O2_CTFWRITER_SPEC
#define O2_CTFWRITER_SPEC

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "DataFormatsParameters/GRPObject.h"

namespace o2
{
namespace ctf
{

using DetID = o2::detectors::DetID;

class CTFWriterSpec : public o2::framework::Task
{
 public:
  CTFWriterSpec(DetID::mask_t dm, uint64_t r) : mDets(dm), mRun(r) {}
  ~CTFWriterSpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  bool isPresent(DetID id) const { return mDets[id]; }

 private:
  DetID::mask_t mDets; // detectors
  uint64_t mRun = 0;
};

/// create a processor spec
framework::DataProcessorSpec getCTFWriterSpec(DetID::mask_t dets, uint64_t run);

} // namespace ctf
} // namespace o2

#endif /* O2_CTFWRITER_SPEC */
