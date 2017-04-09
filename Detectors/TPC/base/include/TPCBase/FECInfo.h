#ifndef AliceO2_TPC_FECInfo_H
#define AliceO2_TPC_FECInfo_H

#include <iostream>

namespace o2 {
namespace TPC {

class FECInfo {
  public:
    FECInfo() {}
    FECInfo(unsigned char index,
            //unsigned char connector,
            //unsigned char channel,
            unsigned char sampaChip,
            unsigned char sampaChannel)
    : mIndex(index), /*mConnector(connector), mChannel(channel),*/ mSampaChip(sampaChip), mSampaChannel(sampaChannel)
    {}

    const unsigned char getIndex()        const { return mIndex;       }
    //const unsigned char getConnector()    const { return mConnector;   } // -> can be calculated from mSampaChannel and mSampaChip
    //const unsigned char getChannel()      const { return mChannel;     } // -> can be calculated from mSampaChannel and mSampaChip
    const unsigned char getSampaChip()    const { return mSampaChip;   }
    const unsigned char getSampaChannel() const { return mSampaChannel;}

    /// equal operator
    bool operator==(const FECInfo &other) const {
      return mIndex==other.mIndex &&
           (mSampaChip==other.mSampaChip && mSampaChannel==other.mSampaChannel) ;
        //( (mConnector==other.mConnector && mChannel==other.mChannel)
          //|| (mSampaChip==other.mSampaChip && mSampaChannel==other.mSampaChannel) );
    }

    static constexpr int globalSAMPAId(const int fecInSector, const int sampaOnFEC, const int channelOnSAMPA)
    {
      // channelOnSAMPA 0-31, 5 bits
      // sampaOnFEC     0- 4, 3 bits
      // fecInSector    0-90, 7 bits
      return (fecInSector<<8) + (sampaOnFEC<<5) + channelOnSAMPA;
    }

    static constexpr int fecInSector   (const int globalSAMPAId) { return globalSAMPAId >> 8; }
    static constexpr int sampaOnFEC    (const int globalSAMPAId)  { return (globalSAMPAId >> 5) & 7; }
    static constexpr int channelOnSAMPA(const int globalSAMPAId) { return globalSAMPAId & 31; }

    static void sampaInfo(const int globalSAMPAId, int& fecInSector, int& sampaOnFEC, int& channelOnSAMPA)
    {
      fecInSector    = (globalSAMPAId >> 8);
      sampaOnFEC     = (globalSAMPAId >> 5) & 7;
      channelOnSAMPA = (globalSAMPAId & 31);
    }

    /// smaller operator
    bool operator<(const FECInfo &other) const
    {
      if (mIndex < other.mIndex) { return true; }
      if (mSampaChip < other.mSampaChip) { return true; }
      if (mSampaChip==other.mSampaChip && mSampaChannel<other.mSampaChannel) { return true; }
      std::cout << *this << " < " << other << std::endl;
      return false;
    }

    std::ostream& print(std::ostream& out = std::cout) const
    {
      out << "FEC in sector [" << int(mIndex)
          //<<"], FEC connector [" << int(mConnector) << "], FEC channel [" << int(mChannel)
          << "], SAMPA chip [" << int(mSampaChip) << "], SAMPA channel [" << int(mSampaChannel) << "]";
      return out;
    }

    friend std::ostream& operator<< (std::ostream& out, const FECInfo& fec)
    {
      fec.print(out);
      return out;
    }

  private:
    unsigned char mIndex        {0};   ///< FEC number in the sector
    unsigned char mSampaChip    {0};   ///< SAMPA chip on the FEC
    unsigned char mSampaChannel {0};   ///< Cannel on the SAMPA chip

    //unsigned char mConnector    {0};   ///< Connector on the FEC -> Can be deduced from mSampaChip and mSampaChannel
    //unsigned char mChannel      {0};   ///< Channel on the FEC -> Can be deduced from mSampaChip and mSampaChannel
};

}
}

#endif
