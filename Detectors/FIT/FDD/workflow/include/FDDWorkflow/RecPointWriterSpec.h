// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RecPointWriterSpec.h

#ifndef O2_FDD_RECPOINTWRITER_H
#define O2_FDD_RECPOINTWRITER_H

#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

namespace o2
{
namespace fdd
{

/// create a processor spec
/// write ITS clusters a root file
framework::DataProcessorSpec getFDDRecPointWriterSpec(bool useMC);

} // namespace fdd
} // namespace o2

#endif /* O2_FDD_RECPOINTWRITER_H */
