// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReaderSpec.h
/// @author Matthias Richter
/// @since  2018-03-23
/// @brief  Processor spec for a reader of TPC data from ROOT file

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace TPC
{

/// create a processor spec
/// read simulated TPC clusters from file and publish
framework::DataProcessorSpec getDigitReaderSpec();

} // end namespace TPC
} // end namespace o2
