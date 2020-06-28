// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUReconstructionIncludesEC0.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONINCLDUESEC0_H
#define GPURECONSTRUCTIONINCLDUESEC0_H

#if defined(HAVE_O2HEADERS) && !defined(GPUCA_NO_EC0_TRAITS)
#include "EC0tracking/TrackerTraitsEC0CPU.h"
#include "EC0tracking/VertexerTraitsEC0.h"
#else
namespace o2
{
namespace ecl
{
class TrackerTraitsEC0
{
};
class TrackerTraitsEC0CPU : public TrackerTraitsEC0
{
};
class VertexerTraitsEC0
{
};
} // namespace ecl
} // namespace o2
#if defined(HAVE_O2HEADERS)
#include "EC0tracking/Road.h"
#include "EC0tracking/Cluster.h"
#endif
#endif

#endif
