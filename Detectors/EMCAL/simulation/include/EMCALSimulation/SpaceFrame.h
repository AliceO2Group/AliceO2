// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
////////////////////////////////////////////////////////////////////////////
///
/// \class  SpaceFrame
/// \brief Space Frame implementation
///
/// EMCAL Space Frame implementation (converted from AliRoot)
///
/// \author Ryan M. Ward, <rmward@calpoly.edu>, Cal Poly
///
////////////////////////////////////////////////////////////////////////////
class SpaceFrame
{
 public:
  ///
  /// Default constructor. Initialize parameters
  SpaceFrame() = default;

  ///
  /// Destructor
  virtual ~SpaceFrame() = default;

  ///
  /// This method assembles the Geometries and places them into the
  /// Alice master volume
  void CreateGeometry();
};
} // namespace emcal
} // namespace o2

#endif // ALIEMCALSPACEFRAME_H
