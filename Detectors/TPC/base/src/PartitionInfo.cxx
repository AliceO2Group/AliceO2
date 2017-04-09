#include "TPCBase/PartitionInfo.h"

namespace o2 {
namespace TPC {

PartitionInfo::PartitionInfo(const unsigned char  numberOfFECs,
                             const unsigned char  sectorFECOffset,
                             const unsigned char  numberOfPadRows,
                             const unsigned char  sectorPadRowOffset,
                             const unsigned short numberOfPads
                            )
  : mNumberOfFECs      {numberOfFECs      },
    mSectorFECOffset   {sectorFECOffset   },
    mNumberOfPadRows   {numberOfPadRows   },
    mSectorPadRowOffset{sectorPadRowOffset},
    mNumberOfPads      {numberOfPads      }
{
}

} // namespace TPC
} // namespace AliceO2
