// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_SRC_TRDDIGITIZERSPEC_H_
#define STEER_DIGITIZERWORKFLOW_SRC_TRDDIGITIZERSPEC_H_

namespace o2
{
namespace framework
{
struct DataProcessorSpec;
} // end namespace framework

namespace trd
{

o2::framework::DataProcessorSpec getTRDDigitizerSpec(int channel);

} // end namespace trd
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_SRC_TRDDIGITIZERSPEC_H_ */
