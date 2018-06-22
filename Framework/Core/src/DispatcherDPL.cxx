// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DispatcherDPL.cxx
/// \brief Implementation of DispatcherDPL for O2 Data Sampling
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DispatcherDPL.h"

namespace o2
{
namespace framework
{

DispatcherDPL::DispatcherDPL(const SubSpecificationType dispatcherSubSpec, const QcTaskConfiguration& task,
                             const InfrastructureConfig& cfg)
  : Dispatcher(dispatcherSubSpec, task, cfg)
{
  mDataProcessorSpec.algorithm = AlgorithmSpec{[gen = Dispatcher::BernoulliGenerator(task.fractionOfDataToSample)](
    ProcessingContext & ctx) mutable { processCallback(ctx, gen);
}
};
}

DispatcherDPL::~DispatcherDPL() {}

void DispatcherDPL::processCallback(ProcessingContext& ctx, BernoulliGenerator& bernoulliGenerator)
{
  InputRecord& inputs = ctx.inputs();

  if (bernoulliGenerator.drawLots()) {
    for (auto& input : inputs) {

      Output output = createDispatcherOutput(*input.spec);

      const auto* inputHeader = header::get<header::DataHeader*>(input.header);

      /*if (inputHeader->payloadSerializationMethod == header::gSerializationMethodInvalid) {
        LOG(ERROR) << "DataSampling::dispatcherCallback: input of origin'" << inputHeader->dataOrigin.str
                   << "', description '" << inputHeader->dataDescription.str
                   << "' has gSerializationMethodInvalid.";
      } else*/ if (inputHeader->payloadSerializationMethod == header::gSerializationMethodROOT) {
        ctx.outputs().adopt(output, DataRefUtils::as<TObject>(input).release());
      } else { // POD
        // todo: do it non-copy, when API is available
        auto outputMessage = ctx.outputs().make<char>(output, inputHeader->payloadSize);
        const char* inputPayload = input.payload;
        for (char& it : outputMessage) {
          it = *inputPayload++;
        }
      }

      LOG(DEBUG) << "DataSampler sends data from subspec " << input.spec->subSpec;
    }
  }
}

void DispatcherDPL::addSource(const DataProcessorSpec& externalDataProcessor, const OutputSpec& externalOutput,
                              const std::string& binding)
{
  InputSpec newInput{
    binding,
    externalOutput.origin,
    externalOutput.description,
    externalOutput.subSpec,
    externalOutput.lifetime
  };

  mDataProcessorSpec.inputs.push_back(newInput);
  Output newOutput = createDispatcherOutput(newInput);
  OutputSpec spec {
    newOutput.origin,
    newOutput.description,
    newOutput.subSpec,
    newOutput.lifetime
  };
  if (mCfg.enableParallelDispatchers ||
      std::find(mDataProcessorSpec.outputs.begin(), mDataProcessorSpec.outputs.end(), spec) ==
        mDataProcessorSpec.outputs.end()) {

    mDataProcessorSpec.outputs.push_back(spec);
  }

  if (mCfg.enableTimePipeliningDispatchers &&
      mDataProcessorSpec.maxInputTimeslices < externalDataProcessor.maxInputTimeslices) {
    mDataProcessorSpec.maxInputTimeslices = externalDataProcessor.maxInputTimeslices;
  }
}

} // namespace framework
} // namespace o2
