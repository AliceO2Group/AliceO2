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

#include <Rtypes.h>
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
  static constexpr short kNCHANNELS = kNumberOfCPVPadsPhi * kNumberOfCPVPadsZ * 3;
  static constexpr float kCPVPadSizePhi = 1.13;
  static constexpr float kCPVPadSizeZ = 2.1093;
  //for hwaddress
  static constexpr short kNPAD = 48;
  static constexpr short kNDilogic = 10;
  static constexpr short kNRow = 16;
  static constexpr short kNMod = 4;

  /// Available numbering schems:
  /// relative pad coordinates
  /// relId[3]={Module, phi col, z row} where Module=2..4, phi col=0..127, z row=0..59
  /// Absolute pad coordunate
  /// absId=0..128*60*3-1=23039
  /// Raw addresses:
  /// DDL corresponds to one module: ddl=Module
  /// each module consist of 16 columns of width 8 pads: row=0..15
  /// Each column consists of 10 dilogics (in z direction) dilogic=0...9
  /// Ecah dilogic contains 8*6 pads: hwaddress=0...48

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
  static short areNeighbours(unsigned short absId1, unsigned short absId2);

  ///
  /// \return AbsId index of the CPV cell
  ///
  /// \param moduleNumber: module number
  /// \param strip: strip number
  //  \param cell: cell in strip number
  ///
  static unsigned short relToAbsId(short moduleNumber, short iphi, short iz);
  static bool absToRelNumbering(unsigned short absId, short* relid);
  static short absIdToModule(unsigned short absId);
  static void absIdToRelPosInModule(unsigned short absId, float& x, float& z);
  static bool relToAbsNumbering(const short* relId, unsigned short& absId);

  static void hwaddressToAbsId(short ddl, short row, short dilogic, short hw, unsigned short& absId);
  static void absIdToHWaddress(unsigned short absId, short& ddl, short& row, short& dilogic, short& hw);

  static unsigned short getTotalNPads() { return kNCHANNELS; }
  static bool IsPadExists(unsigned short absId)
  {
    return absId >= 0 && absId < getTotalNPads();
  } // TODO: evaluate from real geometry

  ClassDefNV(Geometry, 1);
};
} // namespace cpv
} // namespace o2
#endif
