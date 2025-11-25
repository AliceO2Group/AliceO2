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

/// \file   convert_onnx_to_root_serialized.C
/// \brief  Utility functions to be executed as a ROOT macro for uploading ONNX models to CCDB as ROOT serialized objects and vice versa
/// \author Christian Sonnabend <christian.sonnabend@cern.ch>

// Example execution: root -l -b -q '/scratch/csonnabe/MyO2/O2/GPU/GPUTracking/utils/convert_onnx_to_root_serialized.C("/scratch/csonnabe/PhD/jobs/clusterization/NN/output/21082025_smallWindow_clean/SC/training_data_21082025_reco_noise_supressed_p3t6_CoGselected/SC/PbPb_24arp2/0_5/class1/regression/399_noMom/network/net_fp16.onnx", "", 1, 1, "nnCCDBLayerType=FC/nnCCDBWithMomentum=0/inputDType=FP16/nnCCDBInteractionRate=500/outputDType=FP16/nnCCDBEvalType=regression_c1/nnCCDBBeamType=pp/partName=blob/quality=3", 1, 4108971600000, "Users/c/csonnabe/TPC/Clusterization", "model.root")'

#include "ORTRootSerializer.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "TFile.h"
#include <fstream>
#include <stdexcept>

o2::tpc::ORTRootSerializer serializer;

/// Dumps the char* to a .onnx file -> Directly readable by ONNX runtime or Netron
void dumpOnnxToFile(const char* modelBuffer, uint32_t size, const std::string outputPath)
{
  std::ofstream outFile(outputPath, std::ios::binary | std::ios::trunc);
  if (!outFile.is_open()) {
    throw std::runtime_error("Failed to open output ONNX file: " + outputPath);
  }
  outFile.write(modelBuffer, static_cast<std::streamsize>(size));
  if (!outFile) {
    throw std::runtime_error("Failed while writing data to: " + outputPath);
  }
  outFile.close();
}

/// Initialize the serialization from an ONNX file
void readOnnxModelFromFile(const std::string modelPath)
{
  std::ifstream inFile(modelPath, std::ios::binary | std::ios::ate);
  if (!inFile.is_open()) {
    throw std::runtime_error("Could not open input ONNX file " + modelPath);
  }
  std::streamsize size = inFile.tellg();
  std::vector<char> mModelBuffer(size);
  inFile.seekg(0, std::ios::beg);
  if (!inFile.read(mModelBuffer.data(), size)) {
    throw std::runtime_error("Could not read input ONNX file " + modelPath);
  }
  inFile.close();
  serializer.setOnnxModel(mModelBuffer.data(), static_cast<uint32_t>(size));
}

/// Initialize the serialization from a ROOT file
void readRootModelFromFile(const std::string rootFilePath, std::string key)
{
  TFile inRootFile(rootFilePath.c_str());
  if (inRootFile.IsZombie()) {
    throw std::runtime_error("Could not open input ROOT file " + rootFilePath);
  }
  auto* serPtr = inRootFile.Get<o2::tpc::ORTRootSerializer>(key.c_str());
  if (!serPtr) {
    throw std::runtime_error("Could not find " + key + " in ROOT file " + rootFilePath);
  }
  serializer = *serPtr;
  inRootFile.Close();
}

/// Serialize the ONNX model to a ROOT object and store to file
void onnxToRoot(std::string infile, std::string outfile, std::string key)
{
  readOnnxModelFromFile(infile);
  TFile outRootFile(outfile.c_str(), "RECREATE");
  if (outRootFile.IsZombie()) {
    throw std::runtime_error("Could not create output ROOT file " + outfile);
  }
  outRootFile.WriteObject(&serializer, key.c_str());
  outRootFile.Close();
}

/// Deserialize the ONNX model from a ROOT object and store to a .onnx file
void rootToOnnx(std::string infile, std::string outfile, std::string key)
{
  TFile inRootFile(infile.c_str());
  if (inRootFile.IsZombie()) {
    throw std::runtime_error("Could not open input ROOT file " + infile);
  }
  auto* serPtr = inRootFile.Get<o2::tpc::ORTRootSerializer>(key.c_str());
  if (!serPtr) {
    throw std::runtime_error("Could not find " + key + " in ROOT file " + infile);
  }
  serializer = *serPtr;

  std::ofstream outFile(outfile, std::ios::binary | std::ios::trunc);
  if (!outFile.is_open()) {
    throw std::runtime_error("Failed to open output ONNX file: " + outfile);
  }
  outFile.write(serializer.getONNXModel(), static_cast<std::streamsize>(serializer.getONNXModelSize()));
  if (!outFile) {
    throw std::runtime_error("Failed while writing data to: " + outfile);
  }
  outFile.close();

  inRootFile.Close();
}

/// Upload the ONNX model to CCDB from an ONNX file
/// !!! Adjust the metadata, path and validity !!!
void uploadToCCDBFromONNX(std::string onnxFile,
                          const std::map<std::string, std::string>& metadata,
                          // { // some example metadata entries
                          //   "nnCCDBLayerType": "FC",
                          //   "nnCCDBWithMomentum": "0",
                          //   "inputDType": "FP16",
                          //   "nnCCDBInteractionRate": "500",
                          //   "outputDType": "FP16",
                          //   "nnCCDBEvalType": "regression_c1",
                          //   "nnCCDBBeamType": "pp",
                          //   "partName": "blob",
                          //   "quality": "3"
                          // }
                          long tsMin /* = 1 */,
                          long tsMax /* = 4108971600000 */,
                          std::string ccdbPath /* = "Users/c/csonnabe/TPC/Clusterization" */,
                          std::string objname /* = "net_regression_r1.root" */,
                          std::string ccdbUrl /* = "http://alice-ccdb.cern.ch" */)
{
  readOnnxModelFromFile(onnxFile);

  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);

  // build full CCDB path including filename
  const std::string fullPath = ccdbPath; //.back() == '/' ? (ccdbPath + objname) : (ccdbPath + "/" + objname);

  api.storeAsTFileAny(&serializer, fullPath, metadata, tsMin, tsMax);
}

/// Upload the ONNX model to CCDB from a ROOT file
/// !!! Adjust the metadata, path and validity !!!
void uploadToCCDBFromROOT(std::string rootFile,
                          const std::map<std::string, std::string>& metadata,
                          long tsMin /* = 1 */,
                          long tsMax /* = 4108971600000 */,
                          std::string ccdbPath /* = "Users/c/csonnabe/TPC/Clusterization" */,
                          std::string objname /* = "net_regression_r1.root" */,
                          std::string ccdbUrl /* = "http://alice-ccdb.cern.ch" */)
{
  // read ROOT file, extract ORTRootSerializer object and upload via storeAsTFileAny
  TFile inRootFile(rootFile.c_str());
  if (inRootFile.IsZombie()) {
    throw std::runtime_error("Could not open input ROOT file " + rootFile);
  }

  // if objname is empty, fall back to default CCDB object key
  const std::string key = objname.empty() ? o2::ccdb::CcdbApi::CCDBOBJECT_ENTRY : objname;

  auto* serPtr = inRootFile.Get<o2::tpc::ORTRootSerializer>(key.c_str());
  if (!serPtr) {
    inRootFile.Close();
    throw std::runtime_error("Could not find " + key + " in ROOT file " + rootFile);
  }
  serializer = *serPtr;

  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);

  // build full CCDB path including filename
  const std::string fullPath = ccdbPath; //.back() == '/' ? (ccdbPath + objname) : (ccdbPath + "/" + objname);

  api.storeAsTFileAny(&serializer, fullPath, metadata, tsMin, tsMax);

  inRootFile.Close();
}

void convert_onnx_to_root_serialized(const std::string& onnxFile,
                                     const std::string& rootFile,
                                     int mode = 0,
                                     int ccdbUpload = 0,
                                     const std::string& metadataStr = "nnCCDBLayerType=FC/nnCCDBWithMomentum=0/inputDType=FP16/nnCCDBInteractionRate=500/outputDType=FP16/nnCCDBEvalType=regression_c1/nnCCDBBeamType=pp/partName=blob/quality=3",
                                     long tsMin = 1,
                                     long tsMax = 4108971600000,
                                     std::string ccdbPath = "Users/c/csonnabe/TPC/Clusterization",
                                     std::string objname = "net_regression_r1.root",
                                     std::string ccdbUrl = "http://alice-ccdb.cern.ch")
{
  // parse metadataStr of the form key=value/key2=value2/...
  std::map<std::string, std::string> metadata;
  std::size_t start = 0;
  while (start < metadataStr.size()) {
    auto sep = metadataStr.find('/', start);
    auto token = metadataStr.substr(start, sep == std::string::npos ? std::string::npos : sep - start);
    if (!token.empty()) {
      auto eq = token.find('=');
      if (eq != std::string::npos && eq > 0 && eq + 1 < token.size()) {
        metadata.emplace(token.substr(0, eq), token.substr(eq + 1));
      }
    }
    if (sep == std::string::npos) {
      break;
    }
    start = sep + 1;
  }

  if (ccdbUpload == 0) {
    if (mode == 0)
      onnxToRoot(onnxFile, rootFile, o2::ccdb::CcdbApi::CCDBOBJECT_ENTRY);
    else if (mode == 1)
      rootToOnnx(rootFile, onnxFile, o2::ccdb::CcdbApi::CCDBOBJECT_ENTRY);
  } else if (ccdbUpload == 1) {
    if (mode == 0)
      uploadToCCDBFromROOT(rootFile, metadata, tsMin, tsMax, ccdbPath, objname, ccdbUrl);
    else if (mode == 1)
      uploadToCCDBFromONNX(onnxFile, metadata, tsMin, tsMax, ccdbPath, objname, ccdbUrl);
  }
}
