#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

struct Arguments {
  std::string modelPath;
  std::string provider;
  int deviceId = 0;
  size_t expectedInputElements = 0;
  size_t expectedOutputElements = 0;
  bool requireProviderAssignment = true;
};

std::string toLower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool hasProvider(const std::vector<std::string>& providers, const std::string& provider)
{
  return std::find(providers.begin(), providers.end(), provider) != providers.end();
}

std::string join(const std::vector<std::string>& values)
{
  std::ostringstream os;
  for (size_t i = 0; i < values.size(); ++i) {
    os << (i == 0 ? "" : ", ") << values[i];
  }
  return os.str();
}

void usage(const char* argv0)
{
  std::cerr << "usage: " << argv0
            << " --model MODEL.onnx --provider cpu|migraphx|cuda|tensorrt "
               "[--device-id N] [--expected-input-elements N] "
               "[--expected-output-elements N] [--allow-cpu-fallback]\n";
}

Arguments parseArguments(int argc, char** argv)
{
  Arguments args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto needValue = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--model") {
      args.modelPath = needValue("--model");
    } else if (arg == "--provider") {
      args.provider = toLower(needValue("--provider"));
    } else if (arg == "--device-id") {
      args.deviceId = std::stoi(needValue("--device-id"));
    } else if (arg == "--expected-input-elements") {
      args.expectedInputElements = std::stoull(needValue("--expected-input-elements"));
    } else if (arg == "--expected-output-elements") {
      args.expectedOutputElements = std::stoull(needValue("--expected-output-elements"));
    } else if (arg == "--allow-cpu-fallback") {
      args.requireProviderAssignment = false;
    } else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (args.modelPath.empty()) {
    throw std::runtime_error("--model is required");
  }
  if (args.provider != "cpu" && args.provider != "migraphx" && args.provider != "cuda" && args.provider != "tensorrt") {
    throw std::runtime_error("--provider must be one of: cpu, migraphx, cuda, tensorrt");
  }
  return args;
}

std::string ortProviderName(const std::string& provider)
{
  if (provider == "cpu") {
    return "CPUExecutionProvider";
  }
  if (provider == "migraphx") {
    return "MIGraphXExecutionProvider";
  }
  if (provider == "cuda") {
    return "CUDAExecutionProvider";
  }
  if (provider == "tensorrt") {
    return "TensorrtExecutionProvider";
  }
  throw std::runtime_error("unsupported provider: " + provider);
}

void appendProvider(Ort::SessionOptions& options, const Arguments& args)
{
  if (args.provider == "cpu") {
    return;
  }
  if (args.provider == "cuda") {
#ifdef ORT_CUDA_BUILD
    OrtCUDAProviderOptions cudaOptions{};
    cudaOptions.device_id = args.deviceId;
    options.AppendExecutionProvider_CUDA(cudaOptions);
    return;
#else
    throw std::runtime_error("CUDA execution provider support was not enabled at build time");
#endif
  }
  if (args.provider == "migraphx") {
#ifdef ORT_MIGRAPHX_BUILD
    OrtMIGraphXProviderOptions migraphxOptions{};
    migraphxOptions.device_id = args.deviceId;
    migraphxOptions.migraphx_mem_limit = std::numeric_limits<size_t>::max();
    options.AppendExecutionProvider_MIGraphX(migraphxOptions);
    return;
#else
    throw std::runtime_error("MIGraphX execution provider support was not enabled at build time");
#endif
  }
  if (args.provider == "tensorrt") {
#ifdef ORT_TENSORRT_BUILD
    Ort::TensorRTProviderOptions tensorrtOptions;
    tensorrtOptions.Update({{"device_id", std::to_string(args.deviceId)}});
    options.AppendExecutionProvider_TensorRT_V2(*tensorrtOptions);
    return;
#else
    throw std::runtime_error("TensorRT execution provider support was not enabled at build time");
#endif
  }
}

std::vector<int64_t> concreteShape(std::vector<int64_t> shape)
{
  for (auto& dim : shape) {
    if (dim <= 0) {
      dim = 1;
    }
  }
  return shape;
}

size_t elementCount(const std::vector<int64_t>& shape)
{
  if (shape.empty()) {
    return 1;
  }
  return std::accumulate(shape.begin(), shape.end(), size_t{1}, [](size_t product, int64_t dim) {
    if (dim <= 0) {
      throw std::runtime_error("invalid concrete tensor dimension");
    }
    return product * static_cast<size_t>(dim);
  });
}

std::string shapeString(const std::vector<int64_t>& shape)
{
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    os << (i == 0 ? "" : ",") << shape[i];
  }
  os << "]";
  return os.str();
}

bool assignedToProvider(const Ort::Session& session, const std::string& providerName, size_t& assignedNodes)
{
  assignedNodes = 0;
  for (const auto& subgraph : session.GetEpGraphAssignmentInfo()) {
    if (subgraph.GetEpName() == providerName) {
      assignedNodes += subgraph.GetNodes().size();
    }
  }
  return assignedNodes > 0;
}

} // namespace

int main(int argc, char** argv)
{
  try {
    const auto args = parseArguments(argc, argv);
    const auto providerName = ortProviderName(args.provider);
    const auto availableProviders = Ort::GetAvailableProviders();
    if (!hasProvider(availableProviders, providerName)) {
      throw std::runtime_error(providerName + " is not available in this ONNX Runtime build. Available providers: " + join(availableProviders));
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime-ep-inference");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    appendProvider(options, args);

    Ort::Session session(env, args.modelPath.c_str(), options);
    size_t assignedNodes = 0;
    if (args.provider != "cpu" && args.requireProviderAssignment && !assignedToProvider(session, providerName, assignedNodes)) {
      throw std::runtime_error(providerName + " did not receive any graph nodes");
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<std::string> inputNames;
    std::vector<const char*> inputNamePointers;
    std::vector<std::vector<float>> inputBuffers;
    std::vector<Ort::Value> inputValues;
    size_t totalInputElements = 0;

    const size_t inputCount = session.GetInputCount();
    if (inputCount == 0) {
      throw std::runtime_error("model has no inputs");
    }
    inputNames.reserve(inputCount);
    inputNamePointers.reserve(inputCount);
    inputBuffers.reserve(inputCount);
    inputValues.reserve(inputCount);

    for (size_t i = 0; i < inputCount; ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      inputNames.emplace_back(name.get());
      inputNamePointers.push_back(inputNames.back().c_str());

      auto typeInfo = session.GetInputTypeInfo(i);
      if (typeInfo.GetONNXType() != ONNX_TYPE_TENSOR) {
        throw std::runtime_error("input " + inputNames.back() + " is not a tensor");
      }
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      if (tensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("input " + inputNames.back() + " is not a float tensor");
      }

      const auto shape = concreteShape(tensorInfo.GetShape());
      const auto elements = elementCount(shape);
      totalInputElements += elements;
      inputBuffers.emplace_back(elements);
      for (size_t j = 0; j < elements; ++j) {
        inputBuffers.back()[j] = static_cast<float>((static_cast<int>((i + j) % 23) - 11) * 0.03125f);
      }
      inputValues.emplace_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputBuffers.back().data(), elements, shape.data(), shape.size()));
      std::cout << "input[" << i << "] " << inputNames.back() << " shape=" << shapeString(shape)
                << " elements=" << elements << "\n";
    }

    if (args.expectedInputElements != 0 && totalInputElements != args.expectedInputElements) {
      throw std::runtime_error("model input element count is " + std::to_string(totalInputElements) +
                               ", expected " + std::to_string(args.expectedInputElements));
    }

    std::vector<std::string> outputNames;
    std::vector<const char*> outputNamePointers;
    const size_t outputCount = session.GetOutputCount();
    if (outputCount == 0) {
      throw std::runtime_error("model has no outputs");
    }
    outputNames.reserve(outputCount);
    outputNamePointers.reserve(outputCount);
    for (size_t i = 0; i < outputCount; ++i) {
      auto name = session.GetOutputNameAllocated(i, allocator);
      outputNames.emplace_back(name.get());
      outputNamePointers.push_back(outputNames.back().c_str());
    }

    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               inputNamePointers.data(),
                               inputValues.data(),
                               inputValues.size(),
                               outputNamePointers.data(),
                               outputNamePointers.size());

    if (outputs.size() != outputCount) {
      throw std::runtime_error("ONNX Runtime returned an unexpected number of outputs");
    }

    size_t totalOutputElements = 0;
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (!outputs[i].IsTensor()) {
        throw std::runtime_error("output " + outputNames[i] + " is not a tensor");
      }
      auto tensorInfo = outputs[i].GetTensorTypeAndShapeInfo();
      if (tensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("output " + outputNames[i] + " is not a float tensor");
      }
      const auto shape = tensorInfo.GetShape();
      const auto elements = tensorInfo.GetElementCount();
      totalOutputElements += elements;
      const float* data = outputs[i].GetTensorData<float>();
      for (size_t j = 0; j < elements; ++j) {
        if (!std::isfinite(data[j])) {
          throw std::runtime_error("output " + outputNames[i] + " contains a non-finite value");
        }
      }
      std::cout << "output[" << i << "] " << outputNames[i] << " shape=" << shapeString(shape)
                << " elements=" << elements << "\n";
    }

    if (args.expectedOutputElements != 0 && totalOutputElements != args.expectedOutputElements) {
      throw std::runtime_error("model output element count is " + std::to_string(totalOutputElements) +
                               ", expected " + std::to_string(args.expectedOutputElements));
    }

    std::cout << "provider=" << providerName << " assigned_nodes=" << assignedNodes
              << " total_inputs=" << totalInputElements
              << " total_outputs=" << totalOutputElements << "\n";
    return 0;
  } catch (const Ort::Exception& ex) {
    std::cerr << "ONNX Runtime error: " << ex.what() << "\n";
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
  }

  usage(argv[0]);
  return 1;
}
