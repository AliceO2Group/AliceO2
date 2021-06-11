// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_O2CONTROLLABELS_H
#define O2_FRAMEWORK_O2CONTROLLABELS_H

#include "Framework/DataProcessorLabel.h"

namespace o2::framework
{

/// DataProcessorLabels which are recognized by the --o2-control dump tool
/// and influence its output.
namespace ecs
{

// This label makes AliECS templates register raw (proxy) channels in the global
// template space without regard for the avoiding the duplicates (i.e. not adding "-{{ it }}").
// Effectively, it allows us to declare cross-machine channels, e.g. for QC.
const extern DataProcessorLabel uniqueProxyLabel;

} // namespace ecs
} // namespace o2::framework

#endif //O2_FRAMEWORK_O2CONTROLLABELS_H
