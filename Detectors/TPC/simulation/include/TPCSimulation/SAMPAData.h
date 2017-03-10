/// \file SAMPAData.h
/// \brief Class for data from one SAMPA
/// \author Sebastian Klewin
#ifndef ALICE_O2_TPC_SAMPADATA_H_
#define ALICE_O2_TPC_SAMPADATA_H_

#include <vector>
#include <iostream> 
#include "FairLogger.h" 

namespace AliceO2 {
namespace TPC {

/// \class SAMPAData
/// \brief Class to store data from one SAMPA

class SAMPAData {
  public:

    /// Default constructor
    SAMPAData();

    /// Constructor
    /// @param id SAMPA ID
    SAMPAData(int id);

    /// Constructor
    /// @param id SAMPA ID
    /// @param data 32 ADC values
    SAMPAData(int id, std::vector<int>* data);

    /// Destructor
    ~SAMPAData();

    /// Sets ADC value of SAMPA channel
    /// @param channel SAMPA channel
    /// @param value ADC value
    void setChannel(int channel, int value)   { mData[channel] = value; };

    /// Print function
    /// @param output stream to put the SAMPA data on
    /// @return The output stream 
    std::ostream& Print(std::ostream& output) const;
    friend std::ostream& operator<< (std::ostream& out, const SAMPAData& s) { return s.Print(out); }

    bool operator== (const SAMPAData& rhs) const { return mID == rhs.mID && mData == rhs.mData; };

  private:

    int mID;                    ///< SMAPA ID on FEC (0-4)
    std::vector<int> mData;     ///< vector to store data
};
}
}
#endif
