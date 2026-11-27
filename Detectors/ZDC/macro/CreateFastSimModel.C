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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"
#include <TFile.h>
#include <map>
#include <string>
#endif

#include "ZDCSimulation/FastSimModel.h"
#include "ZDCBase/Constants.h"

/// Creates the CCDB object for the ZDC fast simulation: the ONNX model together with the parameters defining its usage.
///
/// The model is expected to have
///  - an input  "cond_phys" [N,8]  : PDG, E, vx, vy, vz, px, py, pz (raw values, particle going to positive z)
///  - an input  "x0"        [N,10] : gaussian noise
///  - an input  "u"         [N,10] : uniform numbers for the stochastic rounding (pass "" if the model has no such input)
///  - an output              [N,10] : number of photons in ZN (common, 1, 2, 3, 4) and ZP (common, 1, 2, 3, 4)
///
/// ccdbHost: "" or "external", "internal", "test", an explicit URL, or "local" to only write the ROOT file
///           (usable with ZDCSimParam.fastSimLocalFile)
///
/// NB: the default cuts below define the domain of validity of the model and must be adapted to the training sample!
void CreateFastSimModel(std::string onnxFile = "zdc_fastsim.onnx",
                        long tmin = 0, long tmax = -1, std::string ccdbHost = "local",
                        std::string version = "",
                        float eMin = 0., float eMax = 1.e9, float absEtaMin = 0., float vtxRMax = 1.e9, float vtxZMax = 1.e9,
                        std::string inputNameConditions = "cond_phys", std::string inputNameNoise = "x0", std::string inputNameUniform = "u", std::string outputName = "x_phys",
                        std::string localFile = "ZDCFastSimModel.root")
{
  o2::zdc::FastSimModel model;
  if (!model.loadONNX(onnxFile)) {
    LOG(fatal) << "Failed to load the ONNX model from " << onnxFile;
  }
  model.version = version;
  model.pdgCodes = {2212, 2112}; // spectator protons and neutrons
  model.eMin = eMin;
  model.eMax = eMax;
  model.absEtaMin = absEtaMin;
  model.vtxRMax = vtxRMax;
  model.vtxZMax = vtxZMax;
  model.inputNameConditions = inputNameConditions;
  model.inputNameNoise = inputNameNoise;
  model.inputNameUniform = inputNameUniform;
  model.outputName = outputName;
  model.print();

  if (ccdbHost == "local") {
    TFile file(localFile.c_str(), "recreate");
    if (file.IsZombie()) {
      LOG(fatal) << "Cannot create " << localFile;
    }
    file.WriteObjectAny(&model, "o2::zdc::FastSimModel", o2::ccdb::CcdbApi::CCDBOBJECT_ENTRY);
    file.Close();
    LOG(info) << "Model written to " << localFile;
    return;
  }

  if (ccdbHost.size() == 0 || ccdbHost == "external") {
    ccdbHost = "http://alice-ccdb.cern.ch:8080";
  } else if (ccdbHost == "internal") {
    ccdbHost = "http://o2-ccdb.internal/";
  } else if (ccdbHost == "test") {
    ccdbHost = "http://ccdb-test.cern.ch:8080";
  }
  o2::ccdb::CcdbApi api;
  std::map<std::string, std::string> metadata;
  metadata["version"] = version;
  api.init(ccdbHost.c_str());
  LOG(info) << "Storing " << o2::zdc::CCDBPathFastSimModel << " to CCDB host " << ccdbHost;
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&model, o2::zdc::CCDBPathFastSimModel, metadata, tmin, tmax);
}
