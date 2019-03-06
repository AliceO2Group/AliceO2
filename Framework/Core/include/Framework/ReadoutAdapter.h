// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef o2_framework_ReadoutAdapter_H_DEFINED
#define o2_framework_ReadoutAdapter_H_DEFINED

#include "ExternalFairMQDeviceProxy.h"

namespace o2
{
namespace framework
{

/// An adapter function for data sent by Readout.  It gets the raw pages as
/// provided by Readout and wraps them in a multipart message where the
/// DataHeader is specified by the OutputSpec, while the DataProcessing header
/// is an enumeration of the pages received.
InjectorFunction readoutAdapter(OutputSpec const& spec);

} // namespace framework
} // namespace o2

#endif // o2_framework_ReadoutAdapter_H_DEFINED
