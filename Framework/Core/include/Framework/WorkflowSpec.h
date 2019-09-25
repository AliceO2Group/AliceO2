// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_WORKFLOWSPEC_H
#define FRAMEWORK_WORKFLOWSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "Framework/AlgorithmSpec.h"

#include <vector>
#include <functional>
#include <cstddef>

namespace o2
{
namespace framework
{
using WorkflowSpec = std::vector<DataProcessorSpec>;

/// The purpose of this helper is to duplicate a DataProcessorSpec @a
/// original as many times as specified in maxIndex and to amend each
/// instance by invoking amendCallback on them with their own @a id.
WorkflowSpec parallel(DataProcessorSpec original,
                      size_t maxIndex,
                      std::function<void(DataProcessorSpec&, size_t id)> amendCallback);

/// The purpose of this helper is to duplicate a sequence of DataProcessorSpec
/// as many times as specified in maxIndex and to amend each instance by invoking
/// amendCallback on them with their own @a id.
WorkflowSpec parallel(WorkflowSpec specs,
                      size_t maxIndex,
                      std::function<void(DataProcessorSpec&, size_t id)> amendCallback);

/// create parallel pipelines of processors from a template sequence for a number of
/// parallel sub specification IDs. The sub specifications are distributed among the
/// pipelines.
/// serves the case where each input id (subspec) corresponds to outputs amended with the
/// same subspec. Two callback functions allow two configure the list of subspecs for the
/// call.
///
/// Schematic workflow illustration:
///            template pipeline                           parallel pipelines
///                                                    ----       -----       ----
///                                                   |  A0|---->|A0 B0|---->|B0   |
///                                                   |    |     |   C0|---->|C0   |
///                                                   |  A1|---->|A1 B1|---->|B1   |
///                                                   |    |     |   C1|---->|C1   |
///                                                    ----       -----       ----
///
///                                                    ----       -----       ----
///       ----       ----       ----                  |  A2|---->|A2 B2|---->|B2   |
///      |   A|---->|A  B|---->|B   |    becomes      |    |     |   C2|---->|C2   |
///      |    |     |   C|---->|C   |    ======>      |  A3|---->|A3 B3|---->|B3   |
///       ----       ----       ----                  |    |     |   C3|---->|C3   |
///                                                    ----       -----       ----
///                                                                 .
///                                                                 .
///                                                    ----       -----       ----
///                                                   |  An|---->|An Bn|---->|Bn   |
///                                                   |    |     |   Cn|---->|Cn   |
///                                                    ----       -----       ----
///
/// @param specs               the template to be multiplied
/// @param nPipelines          number of pipelines
/// @param getNumberOfSubspecs callback function to return the number of subspecs
/// @param getSubSpec          callback function to return the subspecs at index
WorkflowSpec parallelPipeline(const WorkflowSpec& specs,
                              size_t nPipelines,
                              std::function<size_t()> getNumberOfSubspecs,
                              std::function<size_t(size_t)> getSubSpec);

/// The purpose of this helper is to duplicate an InputSpec @a original
/// as many times as specified in maxIndex and to amend each instance
/// by invoking amendCallback on them with their own @a id. This can be
/// used to programmatically create mergers.
Inputs mergeInputs(InputSpec original,
                   size_t maxIndex,
                   std::function<void(InputSpec&, size_t)> amendCallback);

/// The purpose of this helper is to duplicate a list of InputSpec
/// as many times as specified in maxIndex and to amend each instance
/// by invoking amendCallback on them with their own @a id. This can be
/// used to programmatically create mergers.
Inputs mergeInputs(Inputs inputs,
                   size_t maxIndex,
                   std::function<void(InputSpec&, size_t)> amendCallback);

/// The purpose of this helper is to duplicate a DataProcessorSpec @a original
/// as many times as @a count and make sure that each one of those is picking up
/// a different subchannel. All the inputs of this DataProcessorSpec will
/// have to be adapted to produce one set of inputs per subchannel, in a round
/// robin fashion. All the consumers of this DataProcessorSpec will have to connect
/// to each one of the parallel workers or a "TimeMerger" device will have to do that for
/// you.
DataProcessorSpec timePipeline(DataProcessorSpec original,
                               size_t count);

/// The purpose of this helper is to create a query on the data via a properly formatted
/// @a matcher string which describes data in terms of the O2 Data Model descriptor.
///
/// The syntax of the string is the following:
///
/// binding:origin/description/subSpecification%timemodule;...
///
/// Each ; delimits an InputSpec.
std::vector<InputSpec> select(char const* matcher = "");

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_WORKFLOWSPEC_H
