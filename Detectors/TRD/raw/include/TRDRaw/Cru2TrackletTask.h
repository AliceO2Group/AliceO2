// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   Cru2TrackletTask.h
/// @author Sean Murray
/// @brief  TRD cru output data to tracklet task

#ifndef O2_TRD_CRU2TRACKLETTASK
#define O2_TRD_CRU2TRACKLETTASK

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "TRDRaw/Cru2TrackletTranslator.h"
#include <fstream>

using namespace o2::framework;

namespace o2
{
namespace trd
{

class Cru2TrackletTask : public Task
{
 public:
  Cru2TrackletTask() = default;
  ~Cru2TrackletTask() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  Cru2TrackletTranslator mTranslator;
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_CRU2TRACKLETTASK
