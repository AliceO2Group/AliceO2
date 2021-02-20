// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataProcessorSpec.h"
#include "Framework/OutputSpec.h"
#include <string>
#include <vector>

namespace o2
{

namespace cpv
{

using OutputSpec = framework::OutputSpec;

framework::DataProcessorSpec getDigitsReaderSpec(bool propagateMC = true);
framework::DataProcessorSpec getClustersReaderSpec(bool propagateMC = true);

} // namespace cpv
} // end namespace o2
