// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/ColumnDataSpecsUtils.cxx
/// \brief  Utilities for MID Column Data Specs
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 April 2022

#include "MIDWorkflow/ColumnDataSpecsUtils.h"

#include "Framework/DataSpecUtils.h"
#include "Framework/WorkflowSpec.h"
#include "Headers/DataHeader.h"

namespace o2
{
namespace mid
{
namespace specs
{
std::string getString(std::string_view baseName, std::string_view suffix)
{
  return std::string(baseName) + std::string(suffix);
}

std::string getROFBind(std::string_view baseName)
{
  return getString(baseName, "_rof");
}

std::string getLabelsBind(std::string_view baseName)
{
  return getString(baseName, "_labels");
}

std::string getROFDescription(std::string_view description)
{
  return getString(description, "ROF");
}

std::string getLabelsDescription(std::string_view description)
{
  return getString(description, "LABELS");
}

std::string buildSelector(std::string_view bind, std::string_view description, int subSpec = -1)
{
  std::string sbind(bind.data());
  std::string suffix;
  if (subSpec >= 0) {
    sbind += fmt::format("_{}", subSpec);
    suffix += fmt::format("/{}", subSpec);
  }
  return fmt::format("{}:MID/{}{}", sbind, description, suffix);
}

std::string buildSelectors(std::string_view dataBind, std::string_view dataDesc, std::string_view rofDesc, std::string_view labelsDesc, bool useMC, int subSpec = -1)
{
  std::string selector;
  if (!dataDesc.empty()) {
    selector += buildSelector(dataBind, dataDesc, subSpec);
  }
  if (!rofDesc.empty()) {
    if (!selector.empty()) {
      selector += ";";
    }
    selector += buildSelector(getROFBind(dataBind), rofDesc, subSpec);
  }
  if (useMC && !labelsDesc.empty()) {
    if (!selector.empty()) {
      selector += ";";
    }
    selector += buildSelector(getLabelsBind(dataBind), labelsDesc, subSpec);
  }
  return selector;
}

std::vector<framework::InputSpec> buildInputSpecs(std::string_view dataBind, std::string_view dataDesc, std::string_view rofDesc)
{
  std::string selector;
  for (size_t ievt = 0; ievt < NEvTypes; ++ievt) {
    if (!selector.empty()) {
      selector += ";";
    }
    selector += buildSelectors(dataBind, dataDesc, rofDesc, "", false, ievt);
  }
  return framework::select(selector.c_str());
}

std::vector<framework::InputSpec> buildStandardInputSpecs(std::string_view dataBind, std::string_view dataDesc, bool useMC)
{
  return buildStandardInputSpecs(dataBind, dataDesc, getROFDescription(dataDesc), getLabelsDescription(dataDesc), useMC);
}

std::vector<framework::InputSpec> buildStandardInputSpecs(std::string_view dataBind, std::string_view dataDesc, std::string_view rofDesc, std::string_view labelsDesc, bool useMC)
{
  std::string selector = buildSelectors(dataBind, dataDesc, rofDesc, labelsDesc, useMC, 0);
  return framework::select(selector.c_str());
}

std::vector<framework::OutputSpec> buildOutputSpecs(std::string_view bind, std::string_view description)
{
  std::string selector;
  for (size_t ievt = 0; ievt < NEvTypes; ++ievt) {
    if (!selector.empty()) {
      selector += ";";
    }
    selector += buildSelector(bind, description, ievt);
  }
  auto matchers = framework::select(selector.c_str());
  std::vector<framework::OutputSpec> outputSpecs;
  for (auto& matcher : matchers) {
    outputSpecs.emplace_back(framework::DataSpecUtils::asOutputSpec(matcher));
  }
  return outputSpecs;
}

std::vector<framework::OutputSpec> buildStandardOutputSpecs(std::string_view dataBind, std::string_view dataDesc, bool useMC)
{
  auto selector = buildSelectors(dataBind, dataDesc, getROFDescription(dataDesc), getLabelsDescription(dataDesc), useMC, 0);
  auto matchers = framework::select(selector.data());
  std::vector<framework::OutputSpec> outputSpecs;
  for (auto& matcher : matchers) {
    outputSpecs.emplace_back(framework::DataSpecUtils::asOutputSpec(matcher));
  }
  return outputSpecs;
}

std::vector<framework::Output> buildOutputs(std::vector<framework::OutputSpec> outputSpecs)
{
  std::vector<framework::Output> outputs;
  for (auto& outSpec : outputSpecs) {
    auto matcher = framework::DataSpecUtils::asConcreteDataMatcher(outSpec);
    outputs.emplace_back(framework::Output{matcher.origin, matcher.description, matcher.subSpec});
  }
  return outputs;
}

std::array<gsl::span<const ColumnData>, NEvTypes> getData(framework::ProcessingContext& pc, std::string_view dataBind)
{
  std::array<gsl::span<const ColumnData>, 3> data;
  for (size_t ievt = 0; ievt < NEvTypes; ++ievt) {
    data[ievt] = getInput<ColumnData>(pc, dataBind, ievt);
  }

  return data;
}

gsl::span<const ColumnData> getData(framework::ProcessingContext& pc, std::string_view dataBind, EventType eventType)
{
  return getInput<ColumnData>(pc, dataBind.data(), static_cast<int>(eventType));
}

std::array<gsl::span<const ROFRecord>, NEvTypes> getRofs(framework::ProcessingContext& pc, std::string_view dataBind)
{
  std::array<gsl::span<const ROFRecord>, 3> data;
  for (size_t ievt = 0; ievt < NEvTypes; ++ievt) {
    data[ievt] = getInput<ROFRecord>(pc, getROFBind(dataBind).data(), ievt);
  }

  return data;
}

gsl::span<const ROFRecord> getRofs(framework::ProcessingContext& pc, std::string_view dataBind, EventType eventType)
{
  return getInput<ROFRecord>(pc, getROFBind(dataBind).data(), static_cast<int>(eventType));
}

std::unique_ptr<const o2::dataformats::MCTruthContainer<MCLabel>> getLabels(framework::ProcessingContext& pc, std::string_view dataBind, EventType eventType)
{
  return pc.inputs().get<const o2::dataformats::MCTruthContainer<MCLabel>*>(fmt::format("{}_{}", getLabelsBind(dataBind).data(), static_cast<size_t>(eventType)));
}

} // namespace specs
} // namespace mid
} // namespace o2
