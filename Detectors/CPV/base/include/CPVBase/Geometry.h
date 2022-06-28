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
  // for hwaddress
  static constexpr short kNPAD = 48;
  static constexpr short kNDilogic = 4;
  static constexpr short kNGas = 5;
  static constexpr short kNRow = 16;
  static constexpr short kNMod = 4;

  /// Available numbering schems:
  /// relative pad coordinates
  /// relId[3]={Module, phi col, z row} where Module=2..4, phi col=0..127, z row=0..59
  /// Absolute pad coordunate
  /// absId=0..128*60*3-1=23039
  /// Raw addresses:
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
  static bool absToRelNumbering(unsigned short absId, short* relId);
  static short absIdToModule(unsigned short absId);
  static void absIdToRelPosInModule(unsigned short absId, float& x, float& z);
  static bool relToAbsNumbering(const short* relId, unsigned short& absId);

  static bool hwaddressToAbsId(short ccId, short dil, short gas, short pad, unsigned short& absId);
  static bool absIdToHWaddress(unsigned short absId, short& ccId, short& dil, short& gas, short& pad);

  static unsigned short getTotalNPads() { return kNCHANNELS; }
  static bool IsPadExists(unsigned short absId)
  {
    return absId < getTotalNPads();
  } // TODO: evaluate from real geometry

  // Pad map per 3Gassiplex
  // Fixed mapping
  static constexpr short mPadMap[6][8] = {3, 31, 30, 29, 26, 22, 19, 44,
                                          8, 1, 0, 28, 24, 20, 18, 42,
                                          10, 6, 2, 27, 21, 16, 40, 36,
                                          14, 12, 4, 25, 46, 38, 34, 32,
                                          15, 13, 5, 23, 45, 37, 35, 33,
                                          11, 9, 7, 17, 47, 43, 41, 39};
  static constexpr short mPadToZ[48] = {1, 1, 2, 0, 3, 4, 2, 5, 1, 5, 2, 5, 3, 4, 3, 4, 2, 5, 1, 0, 1, 2, 0, 4, 1, 3, 0, 2, 1, 0, 0, 0, 3, 4, 3, 4, 2, 4, 3, 5, 2, 5, 1, 5, 0, 4, 3, 5};
  static constexpr short mPadToPhi[48] = {2, 1, 2, 0, 2, 2, 1, 2, 0, 1, 0, 0, 1, 1, 0, 0, 5, 3, 6, 6, 5, 4, 5, 3, 4, 3, 4, 3, 3, 3, 2, 1, 7, 7, 6, 6, 7, 5, 5, 7, 6, 6, 7, 5, 7, 4, 4, 4};

  ClassDefNV(Geometry, 1);
};
} // namespace cpv
} // namespace o2
#endif
