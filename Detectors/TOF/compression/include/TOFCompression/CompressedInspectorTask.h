// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressedInspectorTask.h
/// @author Roberto Preghenella
/// @since  2020-01-25
/// @brief  TOF compressed data inspector task

#ifndef O2_TOF_COMPRESSEDINSPECTORTASK
#define O2_TOF_COMPRESSEDINSPECTORTASK

#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include <fstream>

class TFile;
class TH1;
class TH2;

using namespace o2::framework;

namespace o2
{
namespace tof
{

class CompressedInspectorTask : public Task
{
 public:
  CompressedInspectorTask() = default;
  ~CompressedInspectorTask() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

  static DataProcessorSpec getSpec();

 private:
  bool mStatus = false;
  TFile* mFile = nullptr;
  std::map<std::string, TH1*> mHistos1D;
  std::map<std::string, TH1*> mHistos2D;
};

} // namespace tof
} // namespace o2

#endif /* O2_TOF_COMPRESSEDINSPECTORTASK */
