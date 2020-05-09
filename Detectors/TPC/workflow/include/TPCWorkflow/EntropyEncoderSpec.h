// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TPC_ENTROPYENCODERSPEC_H
#define O2_TPC_ENTROPYENCODERSPEC_H
/// @file   EntropyEncoderSpec.h
/// @author Michael Lettrich, Matthias Richter
/// @since  2020-01-16
/// @brief  ProcessorSpec for the TPC cluster entropy encoding

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace tpc
{

/// create a processor spec
framework::DataProcessorSpec getEntropyEncoderSpec(bool inputFromFile);

} // end namespace tpc
} // end namespace o2

#endif // O2_TPC_ENTROPYENCODERSPEC_H
