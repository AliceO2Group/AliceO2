// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_DATASAMPLINGREADOUTADAPTER_H
#define ALICEO2_DATASAMPLINGREADOUTADAPTER_H

#include "ExternalFairMQDeviceProxy.h"

namespace o2
{
namespace framework
{

/// An adapter function for data sent by Readout for Data Sampling / Quality control purposes.
/// It adds DataHeader and DataProcessingHeader using information found in received DataBlockHeaderBase
/// header and given OutputSpec. It **IGNORES** SubSpecification in OutputSpec and uses linkID instead.
InjectorFunction dataSamplingReadoutAdapter(OutputSpec const& spec);

} // namespace framework
} // namespace o2

#endif //ALICEO2_DATASAMPLINGREADOUTADAPTER_H
