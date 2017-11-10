// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RootObjectMergerSpec.cxx
/// @author Matthias Richter
/// @since  2017-11-10
/// @brief  Processor spec for a merger for ROOT objects

#include "RootObjectMergerSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Headers/DataHeader.h"
#include "QCCommon/TMessageWrapper.h"
#include "QCMerger/Merger.h"
#include <FairMQLogger.h>
#include <TMessage.h> // object serialization
#include <memory>  // std::unique_ptr
#include <cstring> // memcpy
#include <string>  // std::string
#include <utility> // std::forward
#include <iostream>

using DataProcessorSpec = o2::framework::DataProcessorSpec;
using Inputs = o2::framework::Inputs;
using Outputs = o2::framework::Outputs;
using Options = o2::framework::Options;
using InputSpec = o2::framework::InputSpec;
using OutputSpec = o2::framework::OutputSpec;
using AlgorithmSpec = o2::framework::AlgorithmSpec;
using InitContext = o2::framework::InitContext;
using ProcessingContext = o2::framework::ProcessingContext;
using VariantType = o2::framework::VariantType;

namespace o2 {
namespace qc {

/// create a processor spec for a ROOT object merger
/// processor is interfacing the common merger of the QC module
/// as actual worker class
DataProcessorSpec getRootObjectMergerSpec() {
  using DataHeader = o2::Header::DataHeader;

  // merger instance to be used in the processing function
  auto merger = new Merger(10);

  // set up the processing function
  // using by-copy capture of the worker instance pointer; the actual variable
  // in this function will come out of scope => no by-reference capture
  // FIXME: the object is not cleaned up in the end, framework functionality
  // going to be added soon
  auto processingFct = [merger] (ProcessingContext &pc) {
    for (auto & input : pc.inputs()) {
      auto dh = o2::Header::get<const DataHeader>(input.header);
      std::cout << dh->dataOrigin.str
      << " " << dh->dataDescription.str
      << " " << dh->payloadSize
      << std::endl;
      auto message = std::make_unique<TMessageWrapper>(const_cast<char*>(input.payload),
                                                       dh->payloadSize);
      auto object = reinterpret_cast<TObject*>(message->ReadObject(message->GetClass()));
      if (!object) continue;
      auto merged = merger->mergeObject(object);
      if (!merged) continue;
      std::cout << "Merger: got merged object " << merged->GetTitle() << std::endl;
      merged->Print();
    }
  };

  return {
    "qc_merger",
    {InputSpec{"qc_producer", "QC", "ROOTOBJECT", 0, InputSpec::QA}},
    Outputs{},
    AlgorithmSpec(processingFct)
  };
}

}
}
