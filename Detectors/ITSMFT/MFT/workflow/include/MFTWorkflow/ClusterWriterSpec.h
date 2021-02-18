// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterWriterSpec.h

#ifndef O2_MFT_CLUSTERWRITER_H_
#define O2_MFT_CLUSTERWRITER_H_

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace mft
{

/// create a processor spec
/// write MFT clusters a root file
framework::DataProcessorSpec getClusterWriterSpec(bool useMC);

} // namespace mft
} // namespace o2

#endif /* O2_MFT_CLUSTERWRITER_H */
