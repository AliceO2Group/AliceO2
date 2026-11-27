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

#include "ZDCSimulation/FastSimModel.h"
#include "Framework/Logger.h"
#include <TParticle.h>
#include <algorithm>
#include <cmath>
#include <fstream>

using namespace o2::zdc;

bool FastSimModel::loadONNX(const std::string& fileName)
{
  std::ifstream in(fileName, std::ios::binary | std::ios::ate);
  if (!in.is_open()) {
    LOG(error) << "Cannot open ONNX file " << fileName;
    return false;
  }
  const std::streamsize size = in.tellg();
  if (size <= 0) {
    LOG(error) << "Empty ONNX file " << fileName;
    return false;
  }
  onnx.resize(size);
  in.seekg(0, std::ios::beg);
  if (!in.read(onnx.data(), size)) {
    LOG(error) << "Failed to read ONNX file " << fileName;
    onnx.clear();
    return false;
  }
  return true;
}

bool FastSimModel::accepts(const TParticle& p) const
{
  if (std::find(pdgCodes.begin(), pdgCodes.end(), p.GetPdgCode()) == pdgCodes.end()) {
    return false;
  }
  if (p.Pz() == 0.) {
    return false; // no side can be assigned
  }
  const auto e = p.Energy();
  if (e < eMin || e > eMax) {
    return false;
  }
  if (std::abs(p.Eta()) < absEtaMin) {
    return false;
  }
  if (std::hypot(p.Vx(), p.Vy()) > vtxRMax || std::abs(p.Vz()) > vtxZMax) {
    return false;
  }
  return true;
}

void FastSimModel::print() const
{
  std::string pdgs;
  for (auto c : pdgCodes) {
    pdgs += std::to_string(c) + " ";
  }
  LOG(info) << "ZDC FastSimModel version '" << version << "', ONNX size " << onnx.size() << " bytes";
  LOG(info) << "  inputs: '" << inputNameConditions << "' [N," << NConditions << "], '" << inputNameNoise << "' [N," << NOutputs << "], '" << inputNameUniform << "' [N," << NOutputs
            << "]; output: '" << outputName << "' [N," << NOutputs << "]";
  LOG(info) << "  PDG: " << pdgs << "| E in [" << eMin << ", " << eMax << "] GeV | |eta| > " << absEtaMin
            << " | vtx R < " << vtxRMax << " cm, |z| < " << vtxZMax << " cm";
}
