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

#ifndef STEER_DIGITIZERWORKFLOW_TPCDISTORTIONMAPFILE_H_
#define STEER_DIGITIZERWORKFLOW_TPCDISTORTIONMAPFILE_H_

#include <functional>
#include <string>

namespace o2
{
namespace framework
{
class ProcessingContext;
}
namespace tpc
{
class Digitizer;

/// File with the TPC distortions scaled to the instantaneous luminosity, shared by the digitizers of a job
/// so that the scaled distortions are computed only once.
///
/// The file is keyed on the ETags of the CCDB distortion maps, TPCCorrMap.lumiInst and the mean-luminosity
/// overrides. A file written for other inputs is replaced. A lock file serialises the digitizers, so that
/// one computes and writes while the others wait and then read.
class TPCDistortionMapFile
{
 public:
  explicit TPCDistortionMapFile(std::string path) : mPath(std::move(path)) {}

  /// Give the digitizer the scaled distortions, from the file or, if it has none for these inputs,
  /// by calling compute, which must leave them in the digitizer; they are then written to the file
  void prepare(framework::ProcessingContext& pc, Digitizer& digitizer, float meanLumi, float meanLumiDerivative,
               const std::function<void()>& compute);

 private:
  std::string mPath; ///< path of the file
  std::string mKey;  ///< key of the distortions the digitizer holds
};

} // namespace tpc
} // namespace o2

#endif
