// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef STEER_DIGITIZERWORKFLOW_FV0DIGITIZER_H_
#define STEER_DIGITIZERWORKFLOW_FV0DIGITIZER_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace fv0
{

o2::framework::DataProcessorSpec getFV0DigitizerSpec(int channel);

} // namespace fv0
} // end namespace o2

#endif /* STEER_DIGITIZERWORKFLOW_FITDIGITIZER_H_ */
