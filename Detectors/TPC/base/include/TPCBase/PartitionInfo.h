// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   PartitionInfo.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Information storage class for the different pad regions
///
/// This class stores information of the different partitions
/// like number of FECs, the global row number offset,
/// number of pad rows, number of pads, ...
/// @see TPCBase/Mapper.h
///
/// origin: TPC
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de

#ifndef ALICEO2_TPC_PARTITIONINFO_H_
#define ALICEO2_TPC_PARTITIONINFO_H_

namespace o2
{
namespace tpc
{

class PartitionInfo
{
 public:
  PartitionInfo() = default;

  PartitionInfo(const unsigned char numberOfFECs,
                const unsigned char sectorFECOffset,
                const unsigned char numberOfPadRows,
                const unsigned char globalRowOffset,
                const unsigned short numberOfPads);

  unsigned char getNumberOfFECs() const { return mNumberOfFECs; }
  unsigned char getSectorFECOffset() const { return mSectorFECOffset; }
  unsigned char getNumberOfPadRows() const { return mNumberOfPadRows; }
  unsigned char getGlobalRowOffset() const { return mGlobalRowOffset; }
  unsigned short getNumberOfPads() const { return mNumberOfPads; }

 private:
  unsigned char mNumberOfFECs{0};
  unsigned char mSectorFECOffset{0};
  unsigned char mNumberOfPadRows{0};
  unsigned char mGlobalRowOffset{0};
  unsigned short mNumberOfPads{0};
};

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_PARTITIONINFO_H_
