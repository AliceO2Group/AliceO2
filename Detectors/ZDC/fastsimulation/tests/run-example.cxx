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

///
/// @file   run-examples.cxx
/// @author SwirtaB
///

#include "FastSimulations.h"

#include <iostream>

// Basic test which check if models can be properly loaded and run.
// Models result is being displayed at stdout (to check if it looks correct).
int main()
{
  // Sample particle data
  std::vector<float> particleData = {5.133179999999999836e+02,
                                     1.454299999999999993e-08,
                                     3.650509999999999381e-08,
                                     -2.731009999999999861e-03,
                                     3.545600000000000140e-02,
                                     -5.182060000000000138e-02,
                                     -5.133179999999999836e+02,
                                     0.000000000000000000e+00,
                                     0.000000000000000000e+00};

  // Loading VAE scales
  std::cout << "Loading ONNX model VAE from: " << o2::zdc::fastsim::gZDCModelPath << std::endl;
  auto vaeScales = o2::zdc::fastsim::loadScales(o2::zdc::fastsim::gZDCModelConfig);

  // If vaeScale.has_value() != true error occured during loading scales
  if (!vaeScales.has_value()) {
    std::cout << "error loading vae model scales" << std::endl;
    return 0;
  }
  // Loading actual model and setting scales
  o2::zdc::fastsim::ConditionalModelSimulation onnxVAEDemo(
    o2::zdc::fastsim::gZDCModelPath, vaeScales->first, vaeScales->second, 1.0);
  std::cout << " ONNX VAE model loaded: " << std::endl;

  // Loading SAE scales
  std::cout << "Loading ONNX model SAE from: " << o2::zdc::fastsim::gSAEModelPath << std::endl;
  auto saeScales = o2::zdc::fastsim::loadScales(o2::zdc::fastsim::gSAEModelConfig);

  // If saeScale.has_value() != true error occured during loading scales
  if (!saeScales.has_value()) {
    std::cout << "error loading sae model scales" << std::endl;
    return 0;
  }
  // Loading actual model and setting scales
  o2::zdc::fastsim::ConditionalModelSimulation onnxSAEDemo(
    o2::zdc::fastsim::gSAEModelPath, saeScales->first, saeScales->second, 1.0);
  std::cout << " ONNX SAE model loaded: " << std::endl;

  auto vaeResult = onnxVAEDemo.getChannels(particleData);
  auto saeResult = onnxSAEDemo.getChannels(particleData);

  // Print output
  for (auto& element : vaeResult) {
    std::cout << element << ", ";
  }
  std::cout << std::endl;
  for (auto& element : saeResult) {
    std::cout << element << ", ";
  }

  return 0;
}
