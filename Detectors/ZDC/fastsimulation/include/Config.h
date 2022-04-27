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
/// @file   Config.h
/// @author SwirtaB
///

#ifndef O2_ZDC_FAST_SIMULATION_CONFIG_H
#define O2_ZDC_FAST_SIMULATION_CONFIG_H

#include <string>

namespace o2::zdc::fastsim
{
/**
 * @brief Global paths to models and scales files.
 *
 */

const std::string gZDCModelPath = "share/Detectors/ZDC/fastsimulation/onnx-models/generator.onnx"; // tmp path
const std::string gZDCModelConfig = "share/Detectors/ZDC/fastsimulation/scales/sae_scales.txt";
const std::string gSAEModelPath = "share/Detectors/ZDC/fastsimulation/onnx-models/sae_model.onnx"; // tmp path
const std::string gSAEModelConfig = "share/Detectors/ZDC/fastsimulation/scales/sae_scales.txt";
const std::string gEONModelPath = "share/Detectors/ZDC/fastsimulation/onnx-models/eon_classifier.onnx"; // tmp path
const std::string gEONModelConfig = "share/Detectors/ZDC/fastsimulation/scales/eon_scales.txt";

} // namespace o2::zdc::fastsim
#endif // ZDC_FAST_SIMULATION_CONFIG_H