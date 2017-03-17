/// \file HalfSAMPAData.h
/// \brief Class for data from one half SAMPA
/// \author Sebastian Klewin
#ifndef ALICE_O2_TPC_HALFSAMPADATA_H_
#define ALICE_O2_TPC_HALFSAMPADATA_H_

#include <vector>
#include <iostream> 
#include <iomanip>
#include "FairLogger.h" 

namespace AliceO2 {
namespace TPC {

/// \class HalfSAMPAData
/// \brief Class to store data from one half SAMPA

class HalfSAMPAData {
  public:

    /// Default constructor
    HalfSAMPAData();

    /// Constructor
    /// @param id SAMPA ID
    /// @param low Which half of SAMPA channels, true for low, false for high
    HalfSAMPAData(int id, bool low);

    /// Constructor
    /// @param id SAMPA ID
    /// @param low Which half of SAMPA channels, true for low, false for high
    /// @param data 32 ADC values
    HalfSAMPAData(int id, bool low, std::vector<short>& data);

    /// Destructor
    ~HalfSAMPAData();

    /// Sets ADC value of SAMPA channel
    /// @param channel SAMPA channel
    /// @param value ADC value
    void setChannel(int channel, short value)   { mData[channel] = value; };

    /// Returns ADC value of SAMPA channel
    /// @param channel SAMPA channel
    short getChannel(int channel)   const { return mData[channel]; };
    short operator[] (int index)    const { return mData[index]; };
    short& operator[] (int index)         { return mData[index]; };

    /// Sets SAMPA channel to Low or High
    /// @param val SAMPA channel
    void setLow(bool val) { mLow = val; };

    /// Sets ID of SAMPA
    /// @param val SAMPA ID
    void setID(int val) { mID = val; };

    /// Returns ID of this SAMPA
    /// @return ID of SAMPA
    int getID() const { return mID; };

    /// Resets internal storage
    void reset() { mID = -1; mLow = true; };// mData.clear(); };

    /// Returns reference to data vector
    /// @return Reference to data vector
    std::vector<short>& getData() { return mData; };


    /// Print function
    /// @param output stream to put the SAMPA data on
    /// @return The output stream 
    std::ostream& Print(std::ostream& output) const;
    friend std::ostream& operator<< (std::ostream& out, const HalfSAMPAData& s) { return s.Print(out); }

    bool operator== (const HalfSAMPAData& rhs) const { return mID == rhs.mID && mData == rhs.mData; };

  private:

    int mID;                    ///< SMAPA ID on FEC (0-4)
    bool mLow;                  ///< True for low half of channel, false for high half of channels
    std::vector<short> mData;     ///< vector to store data
};
}
}
#endif
