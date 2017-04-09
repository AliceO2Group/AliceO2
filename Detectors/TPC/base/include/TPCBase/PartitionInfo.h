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

namespace o2 {
namespace TPC {

class PartitionInfo {
  public:
    PartitionInfo() {}

    PartitionInfo(const unsigned char  numberOfFECs,
                  const unsigned char  sectorFECOffset,
                  const unsigned char  numberOfPadRows,
                  const unsigned char  sectorPadRowOffset,
                  const unsigned short numberOfPads
                 );

    unsigned char  getNumberOfFECs()       const { return mNumberOfFECs;       }
    unsigned char  getSectorFECOffset()    const { return mSectorFECOffset;    }
    unsigned char  getNumberOfPadRows()    const { return mNumberOfPadRows;    }
    unsigned char  getSectorPadRowOffset() const { return mSectorPadRowOffset; }
    unsigned short getNumberOfPads()       const { return mNumberOfPads;       }
  private:
    unsigned char  mNumberOfFECs      {0};
    unsigned char  mSectorFECOffset   {0};
    unsigned char  mNumberOfPadRows   {0};
    unsigned char  mSectorPadRowOffset{0};
    unsigned short mNumberOfPads      {0};

};

} // namespace TPC
} // namespace AliceO2

#endif // ALICEO2_TPC_PARTITIONINFO_H_
