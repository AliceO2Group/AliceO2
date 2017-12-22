// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


#include <iostream>
#include <boost/algorithm/string.hpp>
#include <memory>

#include <Framework/InputSpec.h>
#include <Framework/DataProcessorSpec.h>
#include <Framework/DataSampling.h>
#include <Framework/ExternalFairMQDeviceProxy.h>
#include <Framework/RawDeviceService.h>
#include <Framework/SimpleRawDeviceService.h>
#include <Framework/ParallelContext.h>
#include "Framework/runDataProcessing.h"

#include "FairMQDevice.h"
#include "FairMQTransportFactory.h"
//#include "zeromq/FairMQTransportFactoryZMQ.h"
//#include <options/FairMQProgOptions.h>

using namespace o2::framework;
using DataHeader = o2::Header::DataHeader;

size_t collectionChunkSize = 100;
struct FakeCluster {
    float x;
    float y;
    float z;
    float q;
};

//run 'readout.exe file:///home/pkonopka/alice/Readout/configDummy.cfg' to receive data

void defineDataProcessing(std::vector<DataProcessorSpec> &specs)
{
  OutputSpec readoutOutputSpec{o2::Header::gDataOriginITS, o2::Header::gDataDescriptionRawData, 0, OutputSpec::Timeframe};
  DataProcessorSpec readout = specifyExternalFairMQDeviceProxy(
    "readout",
    Outputs{
      readoutOutputSpec
    },
    "type=sub,method=connect,address=tcp://localhost:5558,rateLogging=1",
    incrementalConverter(readoutOutputSpec, 0, 1)
  );

  DataProcessorSpec processingStage{
    "processingStage",
    Inputs{
      {"its-raw", o2::Header::gDataOriginITS, o2::Header::gDataDescriptionRawData, 0, InputSpec::Timeframe}
    },
    Outputs{
      {o2::Header::gDataOriginITS, o2::Header::gDataDescriptionClusters, OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) [](ProcessingContext &ctx) {
        size_t index = ctx.services().get<ParallelContext>().index1D();

        auto ItsClusters = ctx.allocator().make<FakeCluster>(
          OutputSpec{o2::Header::gDataOriginITS, o2::Header::gDataDescriptionClusters, index}, collectionChunkSize);
        int i = 0;
        for(auto& cluster : ItsClusters){
          assert( i < collectionChunkSize);
          cluster.x = i;
          cluster.y = 2*i;
          cluster.z = 3*i-1;
          cluster.q = rand();
          i++;
        }
      }
    }
  };

  DataProcessorSpec sink{
    "sink",
    Inputs{
      {"its-clusters", o2::Header::gDataOriginITS, o2::Header::gDataDescriptionClusters, InputSpec::Timeframe}
    },
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) [](ProcessingContext &ctx) {
        LOG(DEBUG) << "Invoked";
      }
    }
  };

  DataProcessorSpec dplOutput{
    "output",
    Inputs{
      InputSpec{"its-raw", o2::Header::gDataOriginITS, o2::Header::gDataDescriptionRawData, 0, InputSpec::Timeframe}
    },
    {},
    AlgorithmSpec{
      (AlgorithmSpec::InitCallback)[](InitContext &initContext) {

        auto device = initContext.services().get<RawDeviceService>().device();

        return (AlgorithmSpec::ProcessCallback)[device=device](ProcessingContext &ctx) {

          auto input = ctx.inputs().get("its-raw");
          const auto *header = o2::Header::get<DataHeader>(input.header);

          char *p = new char[header->size()];
          memcpy(p, input.payload, header->size());
          FairMQMessagePtr msg(
            device->NewMessage(
              p, header->size(),
              [](void *data, void *hint) { delete[] reinterpret_cast<char *>(data); },
              p)
          );
          int err = device->Send(msg, "dpl-out");
//          LOG(DEBUG) << "device->Send(msg, \"dpl-out\"): " << err << " header->size(): " << header->size();
        };
      }
    }, {
      ConfigParamSpec{
        "channel-config", VariantType::String,
        "name=dpl-out,type=pub,method=connect,address=tcp://localhost:5450,rateLogging=1", {"Out-of-band channel config"}}
    }
  };
  specs.push_back(dplOutput);


  specs.push_back(readout);
  specs.push_back(processingStage);
  specs.push_back(sink);

}

