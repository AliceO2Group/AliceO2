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

#ifndef TRDWORKFLOW_SRC_TRDCONFIGEVENTSPEC_H_
#define TRDWORKFLOW_SRC_TRDCONFIGEVENTSPEC_H_

namespace o2
{
namespace framework
{
struct DataProcessorSpec;
} // end namespace framework

namespace trd
{

o2::framework::DataProcessorSpec getTRDConfigEventWriterSpec();

} // end namespace trd
} // end namespace o2

#endif /* TRDWORKFLOW_SRC_TRDCONFIGEVENTSPEC_H_ */
