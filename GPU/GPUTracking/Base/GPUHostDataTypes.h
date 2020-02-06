// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUHostDataTypes.h
/// \author David Rohr

#ifndef GPUHOSTDATATYPES_H
#define GPUHOSTDATATYPES_H

#include "GPUCommonDef.h"

// These are complex data types wrapped in simple structs, which can be forward declared.
// Structures used on the GPU can have pointers to these wrappers, when the wrappers are forward declared.
// These wrapped complex types are not meant for usage on GPU

#include <vector>
#include <array>
#include <memory>
#include "DataFormatsTPC/Constants.h"

namespace GPUCA_NAMESPACE
{
class MCCompLabel;
namespace dataformats
{
template <typename TruthElement>
class MCTruthContainer;
} // namespace dataformats

namespace gpu
{

struct GPUTPCDigitsMCInput {
  std::array<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*, o2::tpc::Constants::MAXSECTOR> v;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
