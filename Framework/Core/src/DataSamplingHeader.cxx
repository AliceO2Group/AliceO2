// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DataSamplingHeader.cxx
/// \brief An implementation of O2 Data Sampling Header
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#include "Framework/DataSamplingHeader.h"

namespace o2
{
namespace framework
{

// storage for DataSamplingHeader static members
const uint32_t o2::framework::DataSamplingHeader::sVersion = 1;
const o2::header::HeaderType o2::framework::DataSamplingHeader::sHeaderType = header::String2<uint64_t>("DataSamp");
const o2::header::SerializationMethod o2::framework::DataSamplingHeader::sSerializationMethod = o2::header::gSerializationMethodNone;

} // namespace framework
} // namespace o2