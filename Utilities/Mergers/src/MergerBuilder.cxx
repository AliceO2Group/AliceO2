// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MergerBuilder.cxx
/// \brief Definition of MergerBuilder for O2 Mergers
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include <Framework/DeviceSpec.h>
#include "Framework/DataSpecUtils.h"

#include "Mergers/MergerBuilder.h"
#include "Mergers/Merger.h"

using namespace o2::framework;

namespace o2
{
namespace experimental::mergers
{

MergerBuilder::MergerBuilder() : mName("INVALID"),
                                 mInputSpecs{},
                                 mOutputSpec{header::gDataOriginInvalid, header::gDataDescriptionInvalid},
                                 mConfig{}
{
}

void MergerBuilder::setName(std::string name)
{
  mName = name;
}

void MergerBuilder::setTopologyPosition(size_t layer, size_t id)
{
  mLayer = layer;
  mId = id;
}

void MergerBuilder::setInputSpecs(const framework::Inputs& inputs)
{
  mInputSpecs = inputs;
}

void MergerBuilder::setOutputSpec(const framework::OutputSpec& output)
{
  mOutputSpec = output;
  mOutputSpec.binding = {MergerBuilder::mergerOutputBinding()};
}

void MergerBuilder::setConfig(MergerConfig config)
{
  mConfig = config;
}

framework::DataProcessorSpec MergerBuilder::buildSpec()
{
  framework::DataProcessorSpec merger;

  merger.name = mergerIdString() + "-" + mName + std::to_string(mLayer) + "l-" + std::to_string(mId);

  merger.inputs = mInputSpecs;

  merger.outputs.push_back(mOutputSpec);
  framework::DataAllocator::SubSpecificationType subSpec = DataSpecUtils::getOptionalSubSpec(mOutputSpec).value();
  if (DataSpecUtils::validate(mOutputSpec) == false) {
    // inner layer => generate output spec according to scheme
    subSpec = mergerSubSpec(mLayer, mId);
    merger.outputs[0] = OutputSpec{{mergerOutputBinding()},
                                   mergerDataOrigin(),
                                   mergerDataDescription(mName),
                                   subSpec}; // it servers as a unique merger output ID
  } else {
    // last layer
    merger.outputs[0].binding = {mergerOutputBinding()};
  }

  merger.algorithm = framework::adaptFromTask<Merger>(mConfig, subSpec);

  if (mConfig.publicationDecision.value == PublicationDecision::EachNSeconds) {
    merger.inputs.push_back({"timer-publish", "MRGR", mergerDataDescription("timer-" + mName), mergerSubSpec(mLayer, mId), framework::Lifetime::Timer});
    merger.options.push_back({"period-timer-publish", framework::VariantType::Int, static_cast<int>(mConfig.publicationDecision.param * 1000000), {"timer period"}});
  }

  return std::move(merger);
}

void MergerBuilder::customizeInfrastructure(std::vector<framework::CompletionPolicy>& policies)
{
  auto matcher = [](framework::DeviceSpec const& device) {
    return device.name.find(MergerBuilder::mergerIdString()) != std::string::npos;
  };
  auto callback = [](gsl::span<framework::PartRef const> const& inputs) {
    return framework::CompletionPolicy::CompletionOp::Consume;
  };
  framework::CompletionPolicy mergerConsumes{"mergerConsumes", matcher, callback};
  policies.push_back(mergerConsumes);
}

} // namespace experimental::mergers
} // namespace o2
