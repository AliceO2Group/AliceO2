// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionIncludesITS.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONINCLDUESITS_H
#define GPURECONSTRUCTIONINCLDUESITS_H

#if defined(GPUCA_HAVE_O2HEADERS) && !defined(GPUCA_NO_ITS_TRAITS)
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/VertexerTraits.h"
#else
namespace o2
{
namespace its
{
class TrackerTraits
{
};
class TrackerTraitsCPU : public TrackerTraits
{
};
class VertexerTraits
{
};
} // namespace its
} // namespace o2
#if defined(GPUCA_HAVE_O2HEADERS)
#include "ITStracking/Road.h"
#include "ITStracking/Cluster.h"
#endif
#endif

#endif
