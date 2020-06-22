// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   FT0RecPointWriterSpec.h

#ifndef O2_FT0RECPOINTWRITER_H
#define O2_FT0RECPOINTWRITER_H


#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

namespace o2
{
namespace ft0
{

/// create a processor spec
framework::DataProcessorSpec getFT0RecPointWriterSpec(bool useMC);

} // namespace ft0
} // namespace o2

#endif /* O2_FT0RECPOINTWRITER_H */
