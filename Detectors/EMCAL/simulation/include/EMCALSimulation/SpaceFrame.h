// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_SPACEFRAME_H_
#define ALICEO2_EMCAL_SPACEFRAME_H_

#include <Rtypes.h>

namespace o2
{
namespace emcal
{
/// \class  SpaceFrame
/// \brief Space Frame implementation
/// \ingroup EMCALsimulation
/// \author Ryan M. Ward, <rmward@calpoly.edu>, Cal Poly
///
/// EMCAL Space Frame implementation (converted from AliRoot)
class SpaceFrame
{
 public:
  /// \brief Default constructor. Initialize parameters
  SpaceFrame() = default;

  /// \brief Destructor
  virtual ~SpaceFrame() = default;

  /// \brief Assembles the Geometries and places them into the Alice master volume
  void CreateGeometry();
};
} // namespace emcal
} // namespace o2

#endif // ALIEMCALSPACEFRAME_H
