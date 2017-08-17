// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRefUtils.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include "Framework/MetricsService.h"
// FIXME: this should not be needed as the framework should be able to
//        decode TClonesArray by itself.
#include "Framework/TMessageSerializer.h"
#include "FairMQLogger.h"
#include <TClonesArray.h>
#include <TH1F.h>

using namespace o2::framework;
using DataHeader = o2::Header::DataHeader;
using DataOrigin = o2::Header::DataOrigin;
using DataDescription = o2::Header::DataDescription;

using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;

// This is how you can define your processing in a declarative way
void defineDataProcessing(std::vector<DataProcessorSpec> &specs) {
  DataProcessorSpec histogramProducer{
    "producer",
    Inputs{},
    Outputs{
      {"TEST", "HISTOS", OutputSpec::Timeframe},
    },
    [](const std::vector<DataRef> inputs,
      ServiceRegistry& services,
      DataAllocator& allocator) {
      sleep(1);
      // Creates a new message of size 1000 which 
      // has "TPC" as data origin and "CLUSTERS" as data description.
      auto &histoCollection = allocator.newTClonesArray(OutputSpec{"TEST", "HISTOS", 0}, "TH1F", 1000);
      auto histo = new TH1F("h2", "test",100, -10., 10.);
      histo->FillRandom("gaus", 1000);
      histoCollection[0] = histo;
    }
  };

  DataProcessorSpec histogramConsumer{
    "consumer",
    Inputs{
       {"TEST", "HISTOS", InputSpec::Timeframe}
    },
    Outputs{},
    [](const std::vector<DataRef> inputs,
       ServiceRegistry& services,
       DataAllocator& allocator)
    {
      if (inputs.size() != 1) {
        throw std::runtime_error("Expecting one and only one input");
      }
      // FIXME: for the moment we need to do the deserialization ourselves.
      //        this should probably be encoded in the serialization field
      //        of the DataHeader and done automatically by the framework
      auto &ref = inputs.back();
      const DataHeader *header = reinterpret_cast<const DataHeader*>(ref.header);
      // This is actually checked by the framework, so the assert
      // should not trigger, independently of the input.
      assert(header->dataOrigin == DataOrigin("TEST"));
      assert(header->dataDescription == DataDescription("HISTOS"));
      assert(header->payloadSize != 0);

      o2::framework::FairTMessage tm(const_cast<char *>(ref.payload), header->payloadSize);
      auto output = reinterpret_cast<TClonesArray*>(tm.ReadObject(tm.GetClass()));
      if (!output) {
        throw std::runtime_error("Missing output");
      }
      if (output->GetSize() != 1) {
        std::ostringstream str;
        str << "TClonesArray should have only one object, found " << output->GetSize();
        throw std::runtime_error(str.str());
      }
      services.get<MetricsService>().post("histograms/received", output->GetSize());
    }
  };

  specs.push_back(histogramProducer);
  specs.push_back(histogramConsumer);
}
