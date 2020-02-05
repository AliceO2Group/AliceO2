// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CPV_GEOMETRY_H_
#define ALICEO2_CPV_GEOMETRY_H_

#include <string>

#include <RStringView.h>
#include <TMath.h>

namespace o2
{
namespace cpv
{
class Geometry
{
 public:
  static constexpr short kNumberOfCPVPadsPhi = 128;
  static constexpr short kNumberOfCPVPadsZ = 60;
  static constexpr float kCPVPadSizePhi = 1.13;
  static constexpr float kCPVPadSizeZ = 2.1093;

  ///
  /// Default constructor.
  Geometry() = default;

  ///
  /// Copy constructor.
  ///
  Geometry(const Geometry& geom);

  ///
  /// Destructor.
  ///
  ~Geometry() = default;

  ///
  /// Assign operator.
  ///
  Geometry& operator=(const Geometry& rvalue);

  /// \breif Checks if two channels have common side
  /// \param absId1: absId of first channel, order important!
  /// \param absId2: absId of secont channel, order important!
  /// \return  0 are not neighbour but continue searching
  //         = 1 are neighbour
  //         = 2 are not neighbour but do not continue searching
  //         =-1 are not neighbour, continue searching, but do not look before d2 next time
  static int areNeighbours(short absId1, short absId2);

  ///
  /// \return AbsId index of the CPV cell
  ///
  /// \param moduleNumber: module number
  /// \param strip: strip number
  //  \param cell: cell in strip number
  ///
  static short relToAbsId(char moduleNumber, int iphi, int iz);
  static bool absToRelNumbering(short absId, short* relid);
  static char absIdToModule(short absId);
  static void absIdToRelPosInModule(short absId, float& x, float& z);
  static bool relToAbsNumbering(const short* relId, short& absId);

  static int getTotalNPads() { return kNumberOfCPVPadsPhi * kNumberOfCPVPadsZ * 3; }
  static bool IsPadExists(short absId)
  {
    return absId > 0 && absId <= getTotalNPads();
  } // TODO: evaluate from real geometry
};
} // namespace cpv
} // namespace o2
#endif
