// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HalfSAMPAData.h
/// \brief Class for data from one half SAMPA
/// \author Sebastian Klewin
#ifndef ALICE_O2_TPC_HALFSAMPADATA_H_
#define ALICE_O2_TPC_HALFSAMPADATA_H_

#include <array>

namespace o2
{
namespace tpc
{

/// \class HalfSAMPAData
/// \brief Class to store data from one half SAMPA

class HalfSAMPAData
{
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
  HalfSAMPAData(int id, bool low, std::array<short, 16>& data);

  /// Destructor
  ~HalfSAMPAData();

  /// Sets ADC value of SAMPA channel
  /// @param channel SAMPA channel
  /// @param value ADC value
  void setChannel(int channel, short value) { mData[channel] = value; };

  /// Returns ADC value of SAMPA channel
  /// @param channel SAMPA channel
  short getChannel(int channel) const { return mData[channel]; };
  short operator[](int index) const { return mData[index]; };
  short& operator[](int index) { return mData[index]; };

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
  void reset()
  {
    mID = -1;
    mLow = true;
  }; // mData.clear(); };

  /// Returns reference to data
  /// @return Reference to data
  std::array<short, 16>& getData() { return mData; };

  /// Assigns the given value to all elements in the container
  /// @val value the value to assign to the elements
  void fill(short val) { mData.fill(val); };

  /// Print function
  /// @param output stream to put the SAMPA data on
  /// @return The output stream
  std::ostream& Print(std::ostream& output) const;
  friend std::ostream& operator<<(std::ostream& out, const HalfSAMPAData& s) { return s.Print(out); }

  bool operator==(const HalfSAMPAData& rhs) const { return mID == rhs.mID && mLow == rhs.mLow && mData == rhs.mData; };
  bool operator!=(const HalfSAMPAData& rhs) const { return !((*this) == rhs); };

 private:
  int mID;                     ///< SMAPA ID on FEC (0-4)
  bool mLow;                   ///< True for low half of channel, false for high half of channels
  std::array<short, 16> mData; ///< array to store data
};
} // namespace tpc
} // namespace o2
#endif
