#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <boost/thread.hpp>
#include <stdlib.h>
#include <unordered_map>
#include <regex>
#include <chrono>
#include <thread>
#include <iostream>
#include <type_traits>
#include <tuple>
#include <chrono>

#include "Algorithm/RangeTokenizer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/LabelContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "Headers/DataHeader.h"

#include "Steer/MCKinematicsReader.h"

#include "DPLUtils/RootTreeReader.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"

#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"
#include "DataFormatsTPC/Defs.h"

#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCQC/Clusters.h"
#include "TPCBase/Painter.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Mapper.h"

#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CallbacksPolicy.h"

#include "DetectorsRaw/HBFUtils.h"

using namespace o2;
using namespace o2::tpc;
using namespace o2::framework;

namespace o2
{
namespace tpc
{
class onnxGPUinference : public Task
{
  public:

    onnxGPUinference(std::unordered_map<std::string, std::string> options_map) {
        model_path = options_map["path"];
        device = options_map["device"];
        dtype = options_map["dtype"];
        std::stringstream(options_map["device-id"]) >> device_id;
        std::stringstream(options_map["num-iter"]) >> test_size_iter;
        std::stringstream(options_map["execution-threads"]) >> execution_threads;
        std::stringstream(options_map["threads-per-session-cpu"]) >> threads_per_session_cpu;
        std::stringstream(options_map["num-tensors"]) >> test_num_tensors;
        std::stringstream(options_map["size-tensor"]) >> test_size_tensor;
        std::stringstream(options_map["measure-cycle"]) >> epochs_measure;
        std::stringstream(options_map["logging-level"]) >> logging_level;
        std::stringstream(options_map["enable-optimizations"]) >> enable_optimizations;

        LOG(info) << "Options loaded";

        execution_threads = std::min((int)execution_threads, (int)boost::thread::hardware_concurrency());

        // Set the environment variable to use ROCm execution provider
        if(device=="GPU"){
          Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ROCM(session_options, device_id));
          LOG(info) << "ROCM execution provider set";
        } else if(device=="CPU"){
          session_options.SetIntraOpNumThreads(threads_per_session_cpu);
          if(threads_per_session_cpu > 0){
            LOG(info) << "CPU execution provider set with " << threads_per_session_cpu << " threads";
          } else {
            threads_per_session_cpu = 0;
            LOG(info) << "CPU execution provider set with default number of threads";
          }
          if(threads_per_session_cpu > 1){
            session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
          }
        } else {
          LOG(fatal) << "Device not recognized";
        }
        // std::vector<std::string> providers = session.GetProviders();
        // for (const auto& provider : providers) {
        //   LOG(info) << "Using execution provider: " << provider << std::endl;
        // }
        
        if((int)enable_profiling){
          session_options.EnableProfiling((options_map["profiling-output-path"] + "/ORT_LOG_").c_str());
        }
        if(enable_optimizations){
          session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        }
        session_options.SetLogSeverityLevel(logging_level);

        env.resize(execution_threads);
        session.resize(execution_threads);
        for(int s = 0; s < execution_threads; s++){
          env[s] = Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "onnx_model_inference");
          session[s].reset(new Ort::Session{env[s], model_path.c_str(), session_options});
        }
        LOG(info) << "Sessions created";

        LOG(info) << "Number of iterations: " << test_size_iter << ", size of the test tensor: " << test_size_tensor << ", measuring every " << epochs_measure << " cycles, number of tensors: " << test_num_tensors << ", execution threads: " << execution_threads;
      
        for (size_t i = 0; i < session[0]->GetInputCount(); ++i) {
            mInputNames.push_back(session[0]->GetInputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session[0]->GetInputCount(); ++i) {
            mInputShapes.emplace_back(session[0]->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        for (size_t i = 0; i < session[0]->GetOutputCount(); ++i) {
            mOutputNames.push_back(session[0]->GetOutputNameAllocated(i, allocator).get());
        }
        for (size_t i = 0; i < session[0]->GetOutputCount(); ++i) {
            mOutputShapes.emplace_back(session[0]->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        LOG(info) << "Initializing ONNX names and sizes";
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
    };

    void runONNXGPUModel(std::vector<std::vector<Ort::Value>>& input) {
      std::vector<std::thread> threads(execution_threads);
      for (int thrd = 0; thrd < execution_threads; thrd++) {
        threads[thrd] = std::thread([&, thrd] {
          auto outputTensors = session[thrd]->Run(runOptions, inputNamesChar.data(), input[thrd].data(), input[thrd].size(), outputNamesChar.data(), outputNamesChar.size());
        });
      }
      for (auto& thread : threads) {
        thread.join();
      }
    };

    void init(InitContext& ic) final {};
    void run(ProcessingContext& pc) final {
        double time = 0;

        LOG(info) << "Preparing input data";
        // Prepare input data
        std::vector<int64_t> inputShape{test_size_tensor, mInputShapes[0][1]};

        LOG(info) << "Creating memory info";
        Ort::MemoryInfo mem_info("Cpu", OrtAllocatorType::OrtDeviceAllocator, device_id, OrtMemType::OrtMemTypeDefault);

        LOG(info) << "Creating ONNX tensor";
        std::vector<std::vector<Ort::Value>> input_tensor(execution_threads);
        if(dtype=="FP16"){
          std::vector<Ort::Float16_t> input_data(mInputShapes[0][1] * test_size_tensor, (Ort::Float16_t)1.f);  // Example input
          for(int i = 0; i < execution_threads; i++){
            for(int j = 0; j < test_num_tensors; j++){
              input_tensor[i].emplace_back(Ort::Value::CreateTensor<Ort::Float16_t>(mem_info, input_data.data(), input_data.size(), inputShape.data(), inputShape.size()));
            }
          }
        } else {
          std::vector<float> input_data(mInputShapes[0][1] * test_size_tensor, 1.0f);  // Example input
          for(int i = 0; i < execution_threads; i++){
            for(int j = 0; j < test_num_tensors; j++){
              input_tensor[i].emplace_back(Ort::Value::CreateTensor<float>(mem_info, input_data.data(), input_data.size(), inputShape.data(), inputShape.size()));
            }
          }
        }

        LOG(info) << "Starting inference";
        for(int i = 0; i < test_size_iter; i++){
          auto start_network_eval = std::chrono::high_resolution_clock::now();
          runONNXGPUModel(input_tensor);
          // std::vector<float> output = model.inference(test);
          auto end_network_eval = std::chrono::high_resolution_clock::now();
          time += std::chrono::duration<double, std::ratio<1, (unsigned long)1e9>>(end_network_eval - start_network_eval).count();
          if((i % epochs_measure == 0) && (i != 0)){
              time /= 1e9;
              LOG(info) << "Total time: " << time << "s. Timing: " << uint64_t((double)test_size_tensor*epochs_measure*execution_threads*test_num_tensors/time) << " elements / s";
              time = 0;
          }
        }

        // for(auto out : output){
        //   LOG(info) << "Test output: " << out;
        // }
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    };

  private:
    
    std::vector<char> model_buffer;
    std::string model_path, device, dtype;
    int device_id, execution_threads, threads_per_session_cpu, enable_profiling, logging_level, enable_optimizations;
    size_t test_size_iter, test_size_tensor, epochs_measure, test_num_tensors;
    
    Ort::RunOptions runOptions;
    std::vector<Ort::Env> env;
    std::vector<std::shared_ptr<Ort::Session>> session;
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> inputNamesChar, outputNamesChar;
    std::vector<std::string> mInputNames;
    std::vector<std::vector<int64_t>> mInputShapes;
    std::vector<std::string> mOutputNames;
    std::vector<std::vector<int64_t>> mOutputShapes;

    std::string printShape(const std::vector<int64_t>& v)
    {
      std::stringstream ss("");
      for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
      ss << v[v.size() - 1];
      return ss.str();
    };
};
}
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"path", VariantType::String, "./model.pt", {"Path to ONNX model"}},
    {"device", VariantType::String, "CPU", {"Device on which the ONNX model is run"}},
    {"device-id", VariantType::Int, 0, {"Device ID on which the ONNX model is run"}},
    {"dtype", VariantType::String, "-", {"Dtype in which the ONNX model is run (FP16 or FP32)"}},
    {"size-tensor", VariantType::Int, 100, {"Size of the input tensor"}},
    {"execution-threads", VariantType::Int, 1, {"If > 1 will run session->Run() with multiple threads as execution providers"}},
    {"threads-per-session-cpu", VariantType::Int, 0, {"Number of threads per session for CPU execution provider"}},
    {"num-tensors", VariantType::Int, 1, {"Number of tensors on which execution is being performed"}},
    {"num-iter", VariantType::Int, 100, {"Number of iterations"}},
    {"measure-cycle", VariantType::Int, 10, {"Epochs in which to measure"}},
    {"enable-profiling", VariantType::Int, 0, {"Enable profiling"}},
    {"profiling-output-path", VariantType::String, "/scratch/csonnabe/O2_new", {"Path to save profiling output"}},
    {"logging-level", VariantType::Int, 0, {"Logging level"}},
    {"enable-optimizations", VariantType::Int, 0, {"Enable optimizations"}}
  };
  std::swap(workflowOptions, options);
}

// ---------------------------------
#include "Framework/runDataProcessing.h"

DataProcessorSpec testProcess(ConfigContext const& cfgc, std::vector<InputSpec>& inputs, std::vector<OutputSpec>& outputs)
{

  // A copy of the global workflow options from customize() to pass to the task
  std::unordered_map<std::string, std::string> options_map{
    {"path", cfgc.options().get<std::string>("path")},
    {"device", cfgc.options().get<std::string>("device")},
    {"device-id", std::to_string(cfgc.options().get<int>("device-id"))},
    {"dtype", cfgc.options().get<std::string>("dtype")},
    {"size-tensor", std::to_string(cfgc.options().get<int>("size-tensor"))},
    {"execution-threads", std::to_string(cfgc.options().get<int>("execution-threads"))},
    {"threads-per-session-cpu", std::to_string(cfgc.options().get<int>("threads-per-session-cpu"))},
    {"num-tensors", std::to_string(cfgc.options().get<int>("num-tensors"))},
    {"num-iter", std::to_string(cfgc.options().get<int>("num-iter"))},
    {"measure-cycle", std::to_string(cfgc.options().get<int>("measure-cycle"))},
    {"enable-profiling", std::to_string(cfgc.options().get<int>("enable-profiling"))},
    {"profiling-output-path", cfgc.options().get<std::string>("profiling-output-path")},
    {"logging-level", std::to_string(cfgc.options().get<int>("logging-level"))},
    {"enable-optimizations", std::to_string(cfgc.options().get<int>("enable-optimizations"))}
  };

  return DataProcessorSpec{
    "test-onnx-gpu",
    inputs,
    outputs,
    adaptFromTask<onnxGPUinference>(options_map),
    Options{
      {"somethingElse", VariantType::String, "-", {"Something else"}}
    }
  };
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  WorkflowSpec specs;

  static std::vector<InputSpec> inputs;
  static std::vector<OutputSpec> outputs;

  specs.push_back(testProcess(cfgc, inputs, outputs));

  return specs;
}