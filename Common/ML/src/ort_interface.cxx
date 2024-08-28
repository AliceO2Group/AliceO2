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

/// \file     ort_interface.cxx
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
/// \brief   A header library for loading ONNX models and inferencing them on CPU and GPU

#include "ML/ort_interface.h"

namespace o2
{

namespace ml
{

void OrtModel::reset(std::unordered_map<std::string, std::string> optionsMap){
  // Load from options map
  if(!optionsMap.contains("model-path")){
    LOG(fatal) << "(ORT) Model path cannot be empty!";
  }
  modelPath = optionsMap["model-path"];
  device = (optionsMap.contains("device") ? optionsMap["device"] : "cpu");
  dtype = (optionsMap.contains("dtype") ? optionsMap["dtype"] : "float");
  deviceId = (optionsMap.contains("device-id") ? std::stoi(optionsMap["device-id"]) : 0);
  allocateDeviceMemory = (optionsMap.contains("allocate-device-memory") ? std::stoi(optionsMap["allocate-device-memory"]) : 0);
  intraOpNumThreads = (optionsMap.contains("intra-op-num-threads") ?  std::stoi(optionsMap["intra-op-num-threads"]) : 0);
  loggingLevel = (optionsMap.contains("logging-level") ? std::stoi(optionsMap["logging-level"]) : 0);
  enableProfiling = (optionsMap.contains("enable-profiling") ? std::stoi(optionsMap["enable-profiling"]) : 0);
  enableOptimizations = (optionsMap.contains("enable-optimizations") ? std::stoi(optionsMap["enable-optimizations"]) : 0);

  if(device == "rocm") {
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ROCM(sessionOptions, deviceId));
    LOG(info) << "(ORT) ROCM execution provider set";
  } else if(device == "migraphx") {
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(sessionOptions, deviceId));
    LOG(info) << "(ORT) MIGraphX execution provider set";
  }
  if(allocateDeviceMemory){
    memoryInfo = Ort::MemoryInfo("Hip", OrtAllocatorType::OrtDeviceAllocator, deviceId, OrtMemType::OrtMemTypeDefault);
    LOG(info) << "(ORT) Memory info set to on-device memory (HIP)";
  }
#if defined(__CUDACC__)
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, deviceId));
  if(allocateDeviceMemory){
    memoryInfo = Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtDeviceAllocator, deviceId, OrtMemType::OrtMemTypeDefault);
    LOG(info) << "(ORT) Memory info set to on-device memory (CUDA)";
  }
#endif

  if(device == "cpu") {
    sessionOptions.SetIntraOpNumThreads(intraOpNumThreads);
    if(intraOpNumThreads > 1){
      sessionOptions.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    } else if(intraOpNumThreads == 1){
      sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    }
    LOG(info) << "(ORT) CPU execution provider set with " << intraOpNumThreads << " threads";
  }

  sessionOptions.DisableMemPattern();
  sessionOptions.DisableCpuMemArena();

  if(enableProfiling){
    if(optionsMap.contains("profiling-output-path")){
      sessionOptions.EnableProfiling((optionsMap["profiling-output-path"] + "/ORT_LOG_").c_str());
    } else {
      LOG(warning) << "(ORT) If profiling is enabled, optionsMap[\"profiling-output-path\"] should be set. Disabling profiling for now.";
      sessionOptions.DisableProfiling();
    }
  } else {
    sessionOptions.DisableProfiling();
  }
  sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel(enableOptimizations));
  sessionOptions.SetLogSeverityLevel(OrtLoggingLevel(loggingLevel));

  env = std::make_shared<Ort::Env>(OrtLoggingLevel(loggingLevel), (optionsMap["onnx-environment-name"].empty() ? "onnx_model_inference" : optionsMap["onnx-environment-name"].c_str()));
  session.reset(new Ort::Session{*env, modelPath.c_str(), sessionOptions});

  for (size_t i = 0; i < session->GetInputCount(); ++i) {
      mInputNames.push_back(session->GetInputNameAllocated(i, allocator).get());
  }
  for (size_t i = 0; i < session->GetInputCount(); ++i) {
      mInputShapes.emplace_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < session->GetOutputCount(); ++i) {
      mOutputNames.push_back(session->GetOutputNameAllocated(i, allocator).get());
  }
  for (size_t i = 0; i < session->GetOutputCount(); ++i) {
      mOutputShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }

  inputNamesChar.resize(mInputNames.size(), nullptr);
  std::transform(std::begin(mInputNames), std::end(mInputNames), std::begin(inputNamesChar),
      [&](const std::string& str) { return str.c_str(); });
  outputNamesChar.resize(mOutputNames.size(), nullptr);
  std::transform(std::begin(mOutputNames), std::end(mOutputNames), std::begin(outputNamesChar),
      [&](const std::string& str) { return str.c_str(); });

  // Print names
  LOG(info) << "Input Nodes:";
  for (size_t i = 0; i < mInputNames.size(); i++) {
    LOG(info) << "\t" << mInputNames[i] << " : " << printShape(mInputShapes[i]);
  }

  LOG(info) << "Output Nodes:";
  for (size_t i = 0; i < mOutputNames.size(); i++) {
    LOG(info) << "\t" << mOutputNames[i] << " : " << printShape(mOutputShapes[i]);
  }
}

void OrtModel::resetSession() { 
  session.reset(new Ort::Session{*env, modelPath.c_str(), sessionOptions});
}

template<class I, class O>
std::vector<O> OrtModel::v2v(std::vector<I>& input, bool clearInput) {
  if constexpr (std::is_same_v<I,O>){
    return input;
  } else {
    std::vector<O> output(input.size());
    std::transform(std::begin(input), std::end(input), std::begin(output), [](I f) { return O(f); });
    if(clearInput) input.clear();
    return output;
  }
}

template<class I, class O> // class I is the input data type, e.g. float, class O is the output data type, e.g. O2::gpu::OrtDataType::Float16_t from O2/GPU/GPUTracking/ML/convert_float16.h
std::vector<O> OrtModel::inference(std::vector<I>& input){
  std::vector<int64_t> inputShape{input.size() / mInputShapes[0][1], mInputShapes[0][1]};
  std::vector<Ort::Value> inputTensor;
  inputTensor.emplace_back(Ort::Value::CreateTensor<O>(memoryInfo, (v2v<I, O>(input)).data(), input.size(), inputShape.data(), inputShape.size()));
  // input.clear();
  auto outputTensors = session->Run(runOptions, inputNamesChar.data(), inputTensor.data(), inputTensor.size(), outputNamesChar.data(), outputNamesChar.size());
  O* outputValues = outputTensors[0].template GetTensorMutableData<O>();
  outputTensors.clear();
  return std::vector<O>{outputValues, outputValues + input.size() * mOutputShapes[0][1]};
}

template<class I, class O> // class I is the input data type, e.g. float, class O is the output data type, e.g. O2::gpu::OrtDataType::Float16_t from O2/GPU/GPUTracking/ML/convert_float16.h
std::vector<O> OrtModel::inference(std::vector<std::vector<I>>& input){
  std::vector<Ort::Value> inputTensor;
  for(auto i : input){
    std::vector<int64_t> inputShape{i.size() / mInputShapes[0][1], mInputShapes[0][1]};
    inputTensor.emplace_back(Ort::Value::CreateTensor<O>(memoryInfo, (v2v<I, O>(i)).data(), i.size(), inputShape.data(), inputShape.size()));
  }
  // input.clear();
  auto outputTensors = session->Run(runOptions, inputNamesChar.data(), inputTensor.data(), inputTensor.size(), outputNamesChar.data(), outputNamesChar.size());
  O* outputValues = outputTensors[0].template GetTensorMutableData<O>();
  outputTensors.clear();
  return std::vector<O>{outputValues, outputValues + input.size() * mOutputShapes[0][1]};
}

// template<class I, class T, class O> // class I is the input data type, e.g. float, class O is the output data type, e.g. O2::gpu::OrtDataType::Float16_t from O2/GPU/GPUTracking/ML/convert_float16.h
// std::vector<O> OrtModel::inference(std::vector<I>& input){
//   std::vector<int64_t> inputShape{input.size(), mInputShapes[0][1]};
//   std::vector<Ort::Value> inputTensor;
//   inputTensor.emplace_back(Ort::Value::CreateTensor<O>(memoryInfo, (v2v<I, O>(input)).data(), input.size(), inputShape.data(), inputShape.size()));
//   input.clear();
//   auto outputTensors = session->Run(runOptions, inputNamesChar.data(), inputTensor.data(), inputTensor.size(), outputNamesChar.data(), outputNamesChar.size());
//   O* outputValues = outputTensors[0].template GetTensorMutableData<O>();
//   outputTensors.clear();
//   return std::vector<O>{outputValues, outputValues + input.size() * mOutputShapes[0][1]};
// }

std::string OrtModel::printShape(const std::vector<int64_t>& v)
{
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

template std::vector<Ort::Float16_t> OrtModel::v2v<float, Ort::Float16_t>(std::vector<float>&, bool);

template std::vector<Ort::Float16_t> OrtModel::inference<float, Ort::Float16_t>(std::vector<float>&);
template std::vector<Ort::Float16_t> OrtModel::inference<Ort::Float16_t, Ort::Float16_t>(std::vector<Ort::Float16_t>&);

template std::vector<Ort::Float16_t> OrtModel::inference<float, Ort::Float16_t>(std::vector<std::vector<float>>&);
template std::vector<Ort::Float16_t> OrtModel::inference<Ort::Float16_t, Ort::Float16_t>(std::vector<std::vector<Ort::Float16_t>>&);

// template std::vector<OrtDataType::Float16_t> OrtModel::v2v<float, OrtDataType::Float16_t>(std::vector<float>&, bool);
// template <> std::vector<Ort::Float16_t> OrtModel::v2v<OrtDataType::Float16_t, Ort::Float16_t>(std::vector<OrtDataType::Float16_t>& input, bool clearInput) {
//   std::vector<Ort::Float16_t> output(input.size());
//   std::transform(std::begin(input), std::end(input), std::begin(output), [](OrtDataType::Float16_t f) { return Ort::Float16_t::FromBits(f.val); });
//   if(clearInput) input.clear();
//   return output;
// };
// template <> std::vector<OrtDataType::Float16_t> OrtModel::v2v<Ort::Float16_t, OrtDataType::Float16_t>(std::vector<Ort::Float16_t>& input, bool clearInput) {
//   std::vector<OrtDataType::Float16_t> output(input.size());
//   std::transform(std::begin(input), std::end(input), std::begin(output), [](Ort::Float16_t f) { return OrtDataType::Float16_t::FromBits(f.val); });
//   if(clearInput) input.clear();
//   return output;
// };
// template std::vector<Ort::Float16_t> OrtModel::v2v<float, Ort::Float16_t>(std::vector<float>&, bool);
// 
// // template std::vector<OrtDataType::Float16_t> OrtModel::inference<OrtDataType::Float16_t, Ort::Float16_t, OrtDataType::Float16_t>(std::vector<OrtDataType::Float16_t>&);
// // template std::vector<OrtDataType::Float16_t> OrtModel::inference<float, Ort::Float16_t, OrtDataType::Float16_t>(std::vector<float>&);
// // template std::vector<float> OrtModel::inference<float, Ort::Float16_t, float>(std::vector<float>&);
// 
// template <> std::vector<OrtDataType::Float16_t> OrtModel::inference<float, OrtDataType::Float16_t>(std::vector<float>& input){
//   return OrtModel::inference<float, Ort::Float16_t, OrtDataType::Float16_t>(input);
// };
// template <> std::vector<OrtDataType::Float16_t> OrtModel::inference<OrtDataType::Float16_t, OrtDataType::Float16_t>(std::vector<OrtDataType::Float16_t>& input) {
//   return OrtModel::inference<OrtDataType::Float16_t, Ort::Float16_t, OrtDataType::Float16_t>(input);
// };
// 
// template <> std::vector<float> OrtModel::inference<float, OrtDataType::Float16_t, float>(std::vector<float>& input) {
//   return OrtModel::inference<float, Ort::Float16_t, float>(input);
// };

} // namespace ml

} // namespace o2