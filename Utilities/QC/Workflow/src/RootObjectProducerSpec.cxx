// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RootObjectProducerSpec.cxx
/// @author Matthias Richter
/// @since  2017-11-10
/// @brief  Processor spec for a test data producer for ROOT objects

#include "RootObjectProducerSpec.h"
#include "QCProducer/Producer.h"
#include "QCProducer/TH1Producer.h"
#include "QCProducer/TH2Producer.h"
#include "QCProducer/TH3Producer.h"
#include "QCProducer/THnProducer.h"
#include "QCProducer/TreeProducer.h"
#include "Framework/DataProcessorSpec.h"
#include <FairMQLogger.h>
#include <TMessage.h> // object serialization
#include <memory>  // std::unique_ptr
#include <cstring> // memcpy
#include <string>  // std::string
#include <utility> // std::forward
#include <stdexcept> // std::runtime_error
#include <type_traits>  // std::conditional

using DataProcessorSpec = o2::framework::DataProcessorSpec;
using Inputs = o2::framework::Inputs;
//using Outputs = o2::framework::Outputs;
using Options = o2::framework::Options;
using OutputSpec = o2::framework::OutputSpec;
using AlgorithmSpec = o2::framework::AlgorithmSpec;
using InitContext = o2::framework::InitContext;
using ProcessingContext = o2::framework::ProcessingContext;
using VariantType = o2::framework::VariantType;

namespace o2 {
namespace qc {

// the namespace for internal implementations
namespace impl {
  // dummy class with variable number of parameters to allow conditional compilation
  class NeverCalled : public Producer {
  public:
    template<typename... Args>
    NeverCalled(Args&&...) {
      throw std::runtime_error("incorrect number of parameters provided in createProducer call");
    }
    TObject* produceData() const override {return nullptr;}
  };

  // helper trait for conditional compilation depending on number of parameters
  template<typename T, size_t N, typename... Args>
  struct conditional {
    using type = typename std::conditional<sizeof...(Args) == N, T, impl::NeverCalled>::type;
  };

  // create the producer instance for given type
  template<typename T, typename... Args>
  Producer* createProducer(const char* type, Args&&... args) {
    LOG(INFO) << "Producing objects of type " << type;
    return new T(std::forward<Args>(args)...);
  }
}

template<typename... Args>
Producer* createProducer(const char* type, Args&&... args) {
  if (std::string(type).compare("TH1F") == 0) {
    using ProducerType = typename impl::conditional<TH1Producer, 3, Args...>::type;
    return impl::createProducer<ProducerType>(type, std::forward<Args>(args)...);
  } else if (std::string(type).compare("TH2F") == 0) {
    using ProducerType = typename impl::conditional<TH2Producer, 3, Args...>::type;
    return impl::createProducer<ProducerType>(type, std::forward<Args>(args)...);
  } else if (std::string(type).compare("TH3F") == 0) {
    using ProducerType = typename impl::conditional<TH3Producer, 3, Args...>::type;
    return impl::createProducer<ProducerType>(type, std::forward<Args>(args)...);
  } else if (std::string(type).compare("THnF") == 0) {
    using ProducerType = typename impl::conditional<THnProducer, 3, Args...>::type;
    return impl::createProducer<ProducerType>(type, std::forward<Args>(args)...);
  } else if (std::string(type).compare("TTree") == 0) {
    using ProducerType = typename impl::conditional<TreeProducer, 4, Args...>::type;
    return impl::createProducer<ProducerType>(type, std::forward<Args>(args)...);
  }
  LOG(ERROR) << "Unknown type of producer: " << type
             << ". Possible types are TH1F,TH2F,TH3F,THnF,TTree.";
  return nullptr;
}

/// create a processor spec for a test producer
/// the processor is interfacing the test producer classes of the QCProducer
/// module as worker classes.
///
/// The processor spec defines the following options to configure the worker
/// classes:
///   --objectType          Type of object: TH1F, TH2F, TH3F, THnF, TTree
///   --objectName          Name of the produced object
///   --objectTitle         Title of the produced object
///   --nBins               Number of bins for histogram objects
///   --nBranches           Number of branches in tree object
///   --nTreeEntries        Number of entries in tree object
DataProcessorSpec getRootObjectProducerSpec() {
  return {
    "qc_producer",
    Inputs{},
    {
      OutputSpec{"QC", "ROOTOBJECT", 0, OutputSpec::QA}
    },
    AlgorithmSpec{
      [](InitContext &ic) {
        // get the option from the init context
        std::shared_ptr<Producer> producer;
        auto type = ic.options().get<std::string>("objectType");
        auto name = ic.options().get<std::string>("objectName");
        auto title = ic.options().get<std::string>("objectTitle");
        if (type != "TTree") {
          auto nBins = ic.options().get<int>("nBins");
          producer.reset(createProducer(type.c_str(), name.c_str(), title.c_str(), nBins));
        } else {
          auto nBranches = ic.options().get<int>("nBranches");
          auto nTreeEntries = ic.options().get<int>("nTreeEntries");
          producer.reset(createProducer(type.c_str(), name.c_str(), title.c_str(), nBranches, nTreeEntries));
        }

        if (!producer) {
          throw std::runtime_error("failed to create producer instance");
        }

        // set up the processing function
        // using by-copy capture of the worker instance shared pointer
        // the shared pointer makes sure to clean up the instance when the processing
        // function gets out of scope
        auto processingFct = [producer] (ProcessingContext &pc) {
          pc.allocator().adopt(OutputSpec{"QC", "ROOTOBJECT", 0, OutputSpec::QA},
                               producer->produceData());
        };

        // return the actual processing function as a lambda function using variables
        // of the init function
        return processingFct;
      }
    },
    Options{
      {"objectType", VariantType::String, "", {"Type of the produced histogram"}},
      {"objectName", VariantType::String, "", {"Name of the produced histogram"}},
      {"objectTitle", VariantType::String, "", {"Title of the produced histogram"}},
      {"nBins", VariantType::Int, -1, {"Number of bins in histogram"}},
      {"nBranches", VariantType::Int, -1, {"Number of branches in tree"}},
      {"nTreeEntries", VariantType::Int, -1, {"Number of entries in tree"}},
    }
  };
}

}
}
