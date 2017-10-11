// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <Common/SubTimeFrameDataModel.h>
#include <Common/SubTimeFrameDPL.h>

#include <Framework/runDataProcessing.h>
#include <Framework/ExternalFairMQDeviceProxy.h>
#include <FairMQLogger.h>
#include <Headers/HeartbeatFrame.h>

using namespace o2::framework;

using DataHeader = o2::header::DataHeader;
using DataOrigin = o2::header::DataOrigin;

// A simple work flow which takes O2 messages from a SubTimeFrameBuilder device as input

WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{

  auto outspecHbFrame = OutputSpec{ o2::header::gDataOriginFLP,
                                    o2::header::gDataDescriptionRawData };
  auto outspecStfMeta = OutputSpec{ o2::header::gDataOriginFLP,
                                    o2::DataDistribution::gDataDescSubTimeFrame };

  auto inspecHbFrame = InputSpec{ "hbframe",
                                  o2::header::gDataOriginFLP,
                                  o2::header::gDataDescriptionRawData };
  auto inspecStfMeta = InputSpec{ "stfmeta",
                                  o2::header::gDataOriginFLP,
                                  o2::DataDistribution::gDataDescSubTimeFrame };

  WorkflowSpec workflow;

  workflow.emplace_back(
    specifyExternalFairMQDeviceProxy(
      "stf-dpl-source",
      /*
        How to define output spec in an imperative form here?
        SubTimeFrameBuilder pushes O2 messages for the readout data (currently) in tuples:

          Data(x) := <DataHeader { "FLP", "RAWDATA", x}, DATA>

        where subSpec is currently the equipment differentiation recieved from the readout.
        An STF can contain 0..* Data(x) messages (data from the same equipment), as well as
        messages from different equipments. Additionally, StfBuilder might create some
        bookkeeping data that needs to be transported to StfSender unchanged.
       */
      { outspecHbFrame /*, outspecStfMeta*/ },
      // NOTE: make sure to enable DPL when running the SubTimeFrameBuilderDevice
      "type=pair,method=connect,address=ipc:///tmp/stf-builder-dpl-pipe,rateLogging=1",
      o2DataModelAdaptor(outspecHbFrame /* currently unused */, 0, 1)));

  workflow.emplace_back(
    DataProcessorSpec{
      "stf-dpl-sink",
      /*
        InputSpec:
        Sink: how to declare catch all for the DPL-> StfSender sink interface.

        Before an STF is transported to an EPN, StfBuilder needs to acquire all messages
        belonging to the STF.
       */
      Inputs{ inspecHbFrame /*, inspecStfMeta */ },
      {},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {

          // throttle the log
          static thread_local unsigned long lPrint = 0;
          if (lPrint++ % 63 == 0) {

            for (const auto& itInputs : ctx.inputs()) {
              // Retrieving message size from API
              const auto* msgHdr = o2::header::get<o2::header::DataHeader*>(itInputs.header);

              o2::DataDistribution::EquipmentIdentifier lId = *msgHdr;
              LOG(INFO) << "Equipment identifier: " << lId.info() << " Payload size: " << msgHdr->payloadSize;
            }
          }
        } } });

  return workflow;
}
