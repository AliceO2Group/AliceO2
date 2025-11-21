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

/// \file   NeuralNetworkClusterizer.cxx
/// \brief  Fetching neural networks for clusterization from CCDB
/// \author Christian Sonnabend

#include <CommonUtils/StringUtils.h>
#include "TPCCalibration/NeuralNetworkClusterizer.h"

using namespace o2::tpc;

void NeuralNetworkClusterizer::initCcdbApi(std::string url)
{
  ccdbApi.init(url);
}

void NeuralNetworkClusterizer::loadIndividualFromCCDB(std::map<std::string, std::string> settings)
{
  metadata["inputDType"] = settings["inputDType"];
  metadata["outputDType"] = settings["outputDType"];
  metadata["nnCCDBEvalType"] = settings["nnCCDBEvalType"];         // classification_1C, classification_2C, regression_1C, regression_2C
  metadata["nnCCDBWithMomentum"] = settings["nnCCDBWithMomentum"]; // 0, 1 -> Only for regression model
  metadata["nnCCDBLayerType"] = settings["nnCCDBLayerType"];       // FC, CNN
  if (settings["nnCCDBInteractionRate"] != "" && std::stoi(settings["nnCCDBInteractionRate"]) > 0) {
    metadata["nnCCDBInteractionRate"] = settings["nnCCDBInteractionRate"];
  }
  if (settings["nnCCDBBeamType"] != "") {
    metadata["nnCCDBBeamType"] = settings["nnCCDBBeamType"];
  }

  LOG(info) << "(NN CLUS) Retrieving network " << settings["nnCCDBPath"] << " from CCDB (NeuralNetworkClusterizer.cxx)";

  bool retrieveSuccess = ccdbApi.retrieveBlob(settings["nnCCDBPath"], settings["outputFolder"], metadata, 1, false, settings["outputFile"]);
  // headers = ccdbApi.retrieveHeaders(settings["nnPathCCDB"], metadata, 1); // potentially needed to init some local variables

  if (retrieveSuccess) {
    LOG(info) << "Network " << settings["nnCCDBPath"] << " retrieved from CCDB, stored at " << settings["outputFile"];
  } else {
    LOG(error) << "Failed to retrieve network from CCDB";
  }
}
