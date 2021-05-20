// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ZDCRecoWriterDPLSpec.h

#ifndef O2_ZDCRECOTWRITERDPLSPEC_H
#define O2_ZDCRECOTWRITERDPLSPEC_H

#include "Framework/DataProcessorSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsZDC/MCLabel.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

/// create a processor spec
framework::DataProcessorSpec getZDCRecoWriterDPLSpec();

} // namespace zdc
} // namespace o2

#endif /* O2_ZDCRECOTWRITERDPLSPEC_H */
