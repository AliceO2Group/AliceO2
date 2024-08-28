#include <iostream>
#include <vector>
#include <fstream>
#include <thread>

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

#include "ML/ort_interface.h"

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
using namespace o2::ml;
using namespace o2::tpc;
using namespace o2::framework;

namespace o2
{
namespace tpc
{
class onnxInference : public Task
{
	public:

		onnxInference(std::unordered_map<std::string, std::string> optionsMap) {
			options_map = optionsMap;
			models = std::vector<OrtModel>(std::stoi(options_map["execution-threads"]));
			for(int thrd = 0; thrd < std::stoi(options_map["execution-threads"]); thrd++) {
            	models[thrd].init(options_map);
			}
		};

		template<class I, class O>
		void runONNXGPUModel(std::vector<std::vector<I>>& input, int execution_threads) {
			std::vector<std::thread> threads(execution_threads);
			for (int thrd = 0; thrd < execution_threads; thrd++) {
				threads[thrd] = std::thread([&, thrd] {
					auto outputTensors = models[thrd].inference<I, O>(input[thrd]);
				});
			}
			for (auto& thread : threads) {
				thread.join();
			}
		};

		template<class I, class O>
		void runONNXGPUModel(std::vector<std::vector<std::vector<I>>>& input, int execution_threads) {
			std::vector<std::thread> threads(execution_threads);
			for (int thrd = 0; thrd < execution_threads; thrd++) {
				threads[thrd] = std::thread([&, thrd] {
					auto outputTensors = models[thrd].inference<I, O>(input[thrd]);
				});
			}
			for (auto& thread : threads) {
				thread.join();
			}
		};

		void init(InitContext& ic) final {};
		void run(ProcessingContext& pc) final {
			double time = 0;
			int test_size_tensor = std::stoi(options_map["size-tensor"]);
			int epochs_measure = std::stoi(options_map["measure-cycle"]);
			int execution_threads = std::stoi(options_map["execution-threads"]);
			int test_num_tensors = std::stoi(options_map["num-tensors"]);
			int test_size_iter = std::stoi(options_map["num-iter"]);

			LOG(info) << "Preparing input data";
			// Prepare input data
			std::vector<int64_t> inputShape{test_size_tensor, models[0].getNumInputNodes()[0][1]};

			LOG(info) << "Creating ONNX tensor";
			std::vector<std::vector<Ort::Float16_t>> input_tensor(execution_threads);
            std::vector<Ort::Float16_t> input_data(models[0].getNumInputNodes()[0][1] * test_size_tensor, Ort::Float16_t(1.0f));  // Example input
            for(int i = 0; i < execution_threads; i++){
				input_tensor[i] = input_data;
				// input_tensor[i].resize(test_num_tensors);
				// for(int j = 0; j < test_num_tensors; j++){
                // 	input_tensor[i][j] = input_data;
				// }
            }

			LOG(info) << "Starting inference";
			for(int i = 0; i < test_size_iter; i++){
				auto start_network_eval = std::chrono::high_resolution_clock::now();
				runONNXGPUModel<Ort::Float16_t, Ort::Float16_t>(input_tensor, execution_threads);
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
        std::vector<OrtModel> models;
        std::unordered_map<std::string, std::string> options_map;
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
		{"intra-op-num-threads", VariantType::Int, 0, {"Number of threads per session for CPU execution provider"}},
		{"num-tensors", VariantType::Int, 1, {"Number of tensors on which execution is being performed"}},
		{"num-iter", VariantType::Int, 100, {"Number of iterations"}},
		{"measure-cycle", VariantType::Int, 10, {"Epochs in which to measure"}},
		{"enable-profiling", VariantType::Int, 0, {"Enable profiling"}},
		{"profiling-output-path", VariantType::String, "/scratch/csonnabe/O2_new", {"Path to save profiling output"}},
		{"logging-level", VariantType::Int, 1, {"Logging level"}},
		{"enable-optimizations", VariantType::Int, 0, {"Enable optimizations"}},
		{"allocate-device-memory", VariantType::Int, 0, {"Allocate the memory on device"}}
	};
	std::swap(workflowOptions, options);
}

// ---------------------------------
#include "Framework/runDataProcessing.h"

DataProcessorSpec testProcess(ConfigContext const& cfgc, std::vector<InputSpec>& inputs, std::vector<OutputSpec>& outputs)
{

	// A copy of the global workflow options from customize() to pass to the task
	std::unordered_map<std::string, std::string> options_map{
		{"model-path", cfgc.options().get<std::string>("path")},
		{"device", cfgc.options().get<std::string>("device")},
		{"device-id", std::to_string(cfgc.options().get<int>("device-id"))},
		{"dtype", cfgc.options().get<std::string>("dtype")},
		{"size-tensor", std::to_string(cfgc.options().get<int>("size-tensor"))},
		{"intra-op-num-threads", std::to_string(cfgc.options().get<int>("intra-op-num-threads"))},
		{"execution-threads", std::to_string(cfgc.options().get<int>("execution-threads"))},
		{"num-tensors", std::to_string(cfgc.options().get<int>("num-tensors"))},
		{"num-iter", std::to_string(cfgc.options().get<int>("num-iter"))},
		{"measure-cycle", std::to_string(cfgc.options().get<int>("measure-cycle"))},
		{"enable-profiling", std::to_string(cfgc.options().get<int>("enable-profiling"))},
		{"profiling-output-path", cfgc.options().get<std::string>("profiling-output-path")},
		{"logging-level", std::to_string(cfgc.options().get<int>("logging-level"))},
		{"enable-optimizations", std::to_string(cfgc.options().get<int>("enable-optimizations"))},
		{"allocate-device-memory", std::to_string(cfgc.options().get<int>("allocate-device-memory"))}
	};

	return DataProcessorSpec{
		"test-onnx-interface",
		inputs,
		outputs,
		adaptFromTask<onnxInference>(options_map),
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