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

/// \file   MIDWorkflow/ColumnDataSpecsUtils.h
/// \brief  Utilities for MID Column Data Specs
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 April 2022

#ifndef O2_MID_COLUMNDATASPECSUTILS_H
#define O2_MID_COLUMNDATASPECSUTILS_H

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <gsl/span>
#include "fmt/format.h"

#include "Framework/DataAllocator.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ProcessingContext.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/MCLabel.h"

namespace o2
{
namespace mid
{
namespace specs
{

/// Returns the input specs for MID Column Data and corresponding ROFs and labels
/// \param dataBind Data binding name
/// \param dataDesc Input data description
/// \param useMC Builds output specs for labels
/// \return Vector of input specs
std::vector<framework::InputSpec> buildInputSpecs(std::string_view dataBind, std::string_view dataDesc, bool useMC);

/// Returns the input specs for MID Column Data and corresponding ROFs and labels
/// \param dataBind Data binding name
/// \param dataDesc Input data description
/// \param rofDesc Input ROF record description
/// \param labelsDesc Input MC labels description
/// \param useMC Builds output specs for labels
/// \return Vector of input specs
std::vector<framework::InputSpec> buildInputSpecs(std::string_view dataBind, std::string_view dataDesc, std::string_view rofDesc, std::string_view labelsDesc, bool useMC);

/// Returns the output specs for the different event types
/// \param bind Binding name
/// \param description Output data description
/// \return Vector of Output specs
std::vector<framework::OutputSpec> buildOutputSpecs(std::string_view bind, std::string_view description);

/// Returns the output specs for MID Column Data and corresponding ROFs and labels
/// \param dataBind Data binding name
/// \param dataDesc Output data description
/// \param useMC Builds output specs for labels
/// \return Vector of Output specs
std::vector<framework::OutputSpec> buildStandardOutputSpecs(std::string_view dataBind, std::string_view dataDesc, bool useMC);

/// Returns the inputs for the different event types
/// \param pc Processing context
/// \param bind Binding name
/// \return Array of spans
template <typename T>
std::array<gsl::span<const T>, NEvTypes> getInput(framework::ProcessingContext& pc, std::string_view bind)
{
  std::array<gsl::span<const T>, 3> data;
  for (auto const& inputRef : framework::InputRecordWalker(pc.inputs())) {
    auto const* dh = framework::DataRefUtils::getHeader<o2::header::DataHeader*>(inputRef);
    auto subSpecIdx = static_cast<size_t>(dh->subSpecification);
    if (framework::DataRefUtils::match(inputRef, bind.data())) {
      data[subSpecIdx] = pc.inputs().get<gsl::span<T>>(inputRef);
    }
  }
  return data;
}

/// Gets the outputs
/// \param outputSpecs Vector of output specs
/// \return vector of outputs
std::vector<framework::Output> buildOutputs(std::vector<framework::OutputSpec> outputSpecs);

/// Returns the array of Column Data
/// \param pc Processing context
/// \param dataBind Data binding name
/// \return Array of Column Data spans
std::array<gsl::span<const ColumnData>, NEvTypes> getData(framework::ProcessingContext& pc, std::string_view dataBind);

/// Returns the Column Data for the specified event type
/// \param pc Processing context
/// \param dataBind Data binding name
/// \param eventType Event type
/// \return Span of ColumnData
gsl::span<const ColumnData> getData(framework::ProcessingContext& pc, std::string_view dataBind, EventType eventType);

/// Returns the array of ROF records
/// \param pc Processing context
/// \param dataBind Data binding name
/// \return Array of ROF Records spans
std::array<gsl::span<const ROFRecord>, NEvTypes> getRofs(framework::ProcessingContext& pc, std::string_view dataBind);

/// Returns the ROF records for the specified event type
/// \param pc Processing context
/// \param dataBind Data binding name
/// \param eventType Event type
/// \return Span of ROF records
gsl::span<const ROFRecord> getRofs(framework::ProcessingContext& pc, std::string_view dataBind, EventType eventType);

/// Returns the MC labels
/// \param pc Processing context
/// \param dataBind Data binding name
/// \return Pointer to MC labels
std::unique_ptr<const o2::dataformats::MCTruthContainer<MCLabel>> getLabels(framework::ProcessingContext& pc, std::string_view dataBind);

} // namespace specs
} // namespace mid
} // namespace o2

#endif // O2_MID_COLUMNDATASPECSUTILS_H
