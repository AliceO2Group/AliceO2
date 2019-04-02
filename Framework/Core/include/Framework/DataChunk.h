// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATACHUNK_H
#define FRAMEWORK_DATACHUNK_H

#include "MemoryResources/MemoryResources.h"

namespace o2
{
namespace framework
{
// FIXME: make sure that a DataChunk can not be copied or assigned, because the context returns the
// object by reference and we have to make sure that the code is using the reference instead of a copy
using DataChunk = std::vector<char, o2::pmr::polymorphic_allocator<char>>;

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DATACHUNK_H
