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

/// \file FastSim.h
/// \brief ONNX based fast simulation of the ZN and ZP response to spectator nucleons

#ifndef O2_ZDC_FASTSIM_H_
#define O2_ZDC_FASTSIM_H_

#include "ZDCSimulation/FastSimModel.h"
#include <memory>
#include <vector>

namespace o2
{
namespace zdc
{

/// Thin wrapper around the ONNX runtime session evaluating a FastSimModel.
/// The ONNX runtime types are hidden on purpose (pImpl).
class FastSim
{
 public:
  FastSim();
  ~FastSim();
  FastSim(const FastSim&) = delete;
  FastSim& operator=(const FastSim&) = delete;

  /// create the inference session; the model is copied and owned by this object
  void init(const FastSimModel& model);
  bool isInitialized() const { return mImpl != nullptr; }
  const FastSimModel& getModel() const { return mModel; }

  /// Evaluate the model for a batch of particles
  /// \param conditions flat array of nParticles * FastSimModel::NConditions values
  /// \param photons    flat array of nParticles * FastSimModel::NOutputs values (resized by the function)
  void run(const std::vector<float>& conditions, int nParticles, std::vector<float>& photons);

 private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
  FastSimModel mModel;
  std::vector<float> mNoise;
  std::vector<float> mUniform;
};

} // namespace zdc
} // namespace o2

#endif
