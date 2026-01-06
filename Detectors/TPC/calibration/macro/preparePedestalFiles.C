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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <numeric>
#include <vector>
#include <string>
#include <string_view>

#include "TFile.h"
#include "TROOT.h"

#include "TPCBaseRecSim/CDBInterface.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Utils.h"
#include "TPCBase/CRUCalibHelpers.h"
#endif

using namespace o2::tpc;
using namespace o2::tpc::cru_calib_helpers;

/// \param sigmaNoiseROCType can be either one value for all ROC types, or {IROC, OROC}, or {IROC, OROC1, OROC2, OROC3}
/// \param minADCROCType can be either one value for all ROC types, or {IROC, OROC}, or {IROC, OROC1, OROC2, OROC3}
void preparePedestalFiles(const std::string_view pedestalFile, std::string outputDir = "./", std::vector<float> sigmaNoiseROCType = {3}, std::vector<float> minADCROCType = {2}, float pedestalOffset = 0, bool onlyFilled = false, bool maskBad = true, float noisyChannelThreshold = 1.5, float sigmaNoiseNoisyChannels = 4, float badChannelThreshold = 6)
{
  const auto& mapper = Mapper::instance();

  // ===| load noise and pedestal from file |===
  CalDet<float> output("Pedestals");
  const CalDet<float>* calPedestal = nullptr;
  const CalDet<float>* calNoise = nullptr;

  if (pedestalFile.find("cdb") != std::string::npos) {
    auto& cdb = CDBInterface::instance();
    if (pedestalFile.find("cdb-test") == 0) {
      cdb.setURL("http://ccdb-test.cern.ch:8080");
    } else if (pedestalFile.find("cdb-prod") == 0) {
      cdb.setURL("http://alice-ccdb.cern.ch");
    }
    const auto timePos = pedestalFile.find("@");
    if (timePos != std::string_view::npos) {
      std::cout << "set time stamp " << std::stol(pedestalFile.substr(timePos + 1).data()) << "\n";
      cdb.setTimeStamp(std::stol(pedestalFile.substr(timePos + 1).data()));
    }
    calPedestal = &cdb.getPedestals();
    calNoise = &cdb.getNoise();
  } else {
    TFile f(pedestalFile.data());
    gROOT->cd();
    f.GetObject("Pedestals", calPedestal);
    f.GetObject("Noise", calNoise);
  }

  auto pedestalsThreshold = preparePedestalFiles(*calPedestal, *calNoise, sigmaNoiseROCType, minADCROCType, pedestalOffset, onlyFilled, maskBad, noisyChannelThreshold, sigmaNoiseNoisyChannels, badChannelThreshold);

  const auto& pedestals = pedestalsThreshold["Pedestals"];
  const auto& thresholds = pedestalsThreshold["ThresholdMap"];
  const auto& pedestalsPhys = pedestalsThreshold["PedestalsPhys"];
  const auto& thresholdsPhys = pedestalsThreshold["ThresholdMapPhys"];

  auto pedestalValues = getDataMap(pedestals);
  auto thresholdlValues = getDataMap(thresholds);
  auto pedestalValuesPhysics = getDataMap(pedestalsPhys);
  auto thresholdlValuesPhysics = getDataMap(thresholdsPhys);

  // text files
  const auto outFilePedestalTXT(outputDir + "/pedestal_values.txt");
  const auto outFileThresholdTXT(outputDir + "/threshold_values.txt");

  const auto outFilePedestalPhysTXT(outputDir + "/pedestal_values.physics.txt");
  const auto outFileThresholdPhyTXT(outputDir + "/threshold_values.physics.txt");

  writeValues(outFilePedestalTXT, pedestalValues, onlyFilled);
  writeValues(outFileThresholdTXT, thresholdlValues, onlyFilled);

  writeValues(outFilePedestalPhysTXT, pedestalValuesPhysics, onlyFilled);
  writeValues(outFileThresholdPhyTXT, thresholdlValuesPhysics, onlyFilled);

  // root files
  const auto outFilePedestalROOT(outputDir + "/pedestal_values.root");
  const auto outFileThresholdROOT(outputDir + "/threshold_values.root");

  const auto outFilePedestalPhysROOT(outputDir + "/pedestal_values.physics.root");
  const auto outFileThresholdPhyROOT(outputDir + "/threshold_values.physics.root");

  getCalPad(outFilePedestalTXT, outFilePedestalROOT, "Pedestals");
  getCalPad(outFileThresholdTXT, outFileThresholdROOT, "ThresholdMap");

  getCalPad(outFilePedestalPhysTXT, outFilePedestalPhysROOT, "Pedestals");
  getCalPad(outFileThresholdPhyTXT, outFileThresholdPhyROOT, "ThresholdMap");
}
