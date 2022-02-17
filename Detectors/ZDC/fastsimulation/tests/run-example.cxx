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

int main()
{
  std::array<float, 9> rawConditionalInput = {5.133179999999999836e+02,
                                              1.454299999999999993e-08,
                                              3.650509999999999381e-08,
                                              -2.731009999999999861e-03,
                                              3.545600000000000140e-02,
                                              -5.182060000000000138e-02,
                                              -5.133179999999999836e+02,
                                              0.000000000000000000e+00,
                                              0.000000000000000000e+00};

  std::cout << "Loading ONNX model VAE from: " << o2::zdc::fastsim::gZDCModelPath << std::endl;
  auto vaeScales = o2::zdc::fastsim::loadVaeScales(o2::zdc::fastsim::gZDCModelConfig);
  if (!vaeScales.has_value()) {
    std::cout << "error loading vae model scales" << std::endl;
    return 0;
  }
  o2::zdc::fastsim::VAEModelSimulation onnxVAEDemo(vaeScales->first, vaeScales->second, 0.0);
  std::cout << " ONNX VAE model loaded: " << std::endl;

  std::cout << "Loading ONNX model SAE from: " << o2::zdc::fastsim::gSAEModelPath << std::endl;
  auto saeScales = o2::zdc::fastsim::loadSaeScales(o2::zdc::fastsim::gSAEModelConfig);
  if (!saeScales.has_value()) {
    std::cout << "error loading sae model scales" << std::endl;
    return 0;
  }
  o2::zdc::fastsim::SAEModelSimulation onnxSAEDemo(saeScales->first, saeScales->second);
  std::cout << " ONNX SAE model loaded: " << std::endl;

  onnxVAEDemo.setData(rawConditionalInput);
  onnxVAEDemo.run();
  auto vaeResult = onnxVAEDemo.getChannels();

  std::cout << "VAE results:" << std::endl;
  std::cout << vaeResult[0] << ", " << vaeResult[1] << ", " << vaeResult[2] << ", " << vaeResult[3] << ", " << vaeResult[4] << std::endl;

  onnxSAEDemo.setData(rawConditionalInput);
  onnxSAEDemo.run();
  auto saeResult = onnxSAEDemo.getChannels();

  std::cout << "SAE results:" << std::endl;
  std::cout << saeResult[0] << ", " << saeResult[1] << ", " << saeResult[2] << ", " << saeResult[3] << ", " << saeResult[4] << std::endl;

  return 0;
}