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
#ifndef EMCAL_PEDESTAL_PROCESSOR_DATA_H_
#define EMCAL_PEDESTAL_PROCESSOR_DATA_H_
#include <array>
#include <exception>
#include <iosfwd>
#include <string>
#include <tuple>
#include "MathUtils/Utils.h"

#include "Rtypes.h"

namespace o2::emcal
{

/// \class PedestalProcessorData
/// \brief Exchange container between PedestalProcessorDevice and PedestalAggregatorDevice
/// \ingroup EMCALCalib
/// \author Markus Fasel <markus.fasel@cern.ch>, Oak Ridge National Laboratory
/// \since March 22, 2024
///
/// Object containing arrays of stat accumulators that behave like flat profile histograms
/// calculating mean and RMS of a set of ADC values. Corresponding arrays are used for
/// both FEC and LEDMON channels, and in both cases for high and low gain. Distinction between
/// channel and gain type is done via arguments in the fill and get functions, always defining
/// the true cases with LEDMON and low gain.
class PedestalProcessorData
{
 private:
 public:
  /// \class ChannelIndexException
  /// \brief Handling access to invalid channel index (out-of-bounds)
  class ChannelIndexException : public std::exception
  {
   private:
    unsigned short mChannelIndex; ///< Index of the channel raising the exception
    unsigned short mMaxChannels;  ///< Max. number of channels for a given channel type
    std::string mErrorMessage;    ///< Buffer for the error message

   public:
    /// \brief Constructor
    /// \param channelIndex Index raising the exception
    /// \param maxChannels Maximun number of channels for a given type
    ChannelIndexException(unsigned short channelIndex, unsigned short maxChannels) : std::exception(), mChannelIndex(channelIndex), mMaxChannels(maxChannels)
    {
      mErrorMessage = "Channel index " + std::to_string(mChannelIndex) + " not found (max " + std::to_string(mMaxChannels) + ")";
    }

    /// \brief  Destructor
    ~ChannelIndexException() noexcept final = default;

    /// \brief Get error message of the exception
    /// \return Error message
    const char* what() const noexcept final { return mErrorMessage.data(); }

    /// \brief Get channel index raising the exception
    /// \return Channel index
    unsigned short getChannelIndex() const noexcept { return mChannelIndex; }

    /// \brief Get max number for channels for the type the exception was raised
    /// \return Number of channels
    unsigned short getMaxChannels() const noexcept { return mMaxChannels; }
  };

  using ProfileHistFEC = std::array<o2::math_utils::StatAccumulator, 17664>;
  using ProfileHistLEDMON = std::array<o2::math_utils::StatAccumulator, 480>;
  using PedestalValue = std::tuple<double, double>;

  /// \brief Constructor
  PedestalProcessorData() = default;

  /// \brief Destructor
  ~PedestalProcessorData() = default;

  /// \brief Accumulation operator
  /// \param other Object to add to this object
  /// \return This object after accumulation
  ///
  /// Adding stat. accumulators for all channels to this object. The state of this
  /// object is modified.
  PedestalProcessorData& operator+=(const PedestalProcessorData& other);

  /// \brief Fill ADC value for certain channel
  /// \param adc ADC value
  /// \param tower Absolute tower ID
  /// \param lowGain Switch between low and high gain (true = low gain)
  /// \param LEDMON Switch between LEDMON and FEE data (true = LEDMON)
  /// \throw ChannelIndexException for channel index out-of-range
  void fillADC(unsigned short adc, unsigned short tower, bool lowGain, bool LEDMON);

  /// \brief Get mean ADC and RMS for a certain channel
  /// \param tower Absolute tower ID
  /// \param lowGain Switch between low and high gain (true = low gain)
  /// \param LEDMON Switch between LEDMON and FEE data (true = LEDMON)
  /// \return std::tuple with mean and rms of the ADC distribution for the given channl
  /// \throw ChannelIndexException for channel index out-of-range
  PedestalValue getValue(unsigned short tower, bool lowGain, bool LEDMON) const;

  /// \brief Get number of entries for a certain channel
  /// \param tower Absolute tower ID
  /// \param lowGain Switch between low and high gain (true = low gain)
  /// \param LEDMON Switch between LEDMON and FEE data (true = LEDMON)
  /// \return Number of entries
  /// \throw ChannelIndexException for channel index out-of-range
  int getEntriesForChannel(unsigned short tower, bool lowGain, bool LEDMON) const;

  /// \brief Reset object
  ///
  /// Set all stat accumulators to 0.
  void reset();

  /// \brief Provide access to accumulated data for FEC channels
  /// \param lowGain Low gain data
  /// \return Accumulated data for low gain (if lowGain) or high gain
  const ProfileHistFEC& accessFECData(bool lowGain) const { return lowGain ? mDataFECLG : mDataFECHG; }

  /// \brief Provide access to accumulated data for LEDMON channels
  /// \param lowGain Low gain data
  /// \return Accumulated data for low gain (if lowGain) or high gain
  const ProfileHistLEDMON& accessLEDMONData(bool lowGain) const { return lowGain ? mDataLEDMONLG : mDataLEDMONHG; }

 private:
  ProfileHistFEC mDataFECHG;       ///< Profile for FEC channels, high gain
  ProfileHistFEC mDataFECLG;       ///< Profile for FEC channels, low gain
  ProfileHistLEDMON mDataLEDMONHG; ///< Profile for LEDMON channels, high gain
  ProfileHistLEDMON mDataLEDMONLG; ///< Profile for LEDMON channels, low gain

  ClassDefNV(PedestalProcessorData, 1);
};

/// \brief Sum operator for PedestalProcessorData
/// \param lhs Left hand value of the sum operation
/// \param rhs Right hand value of the sum operation
/// \return Sum of the two containers (all channels)
PedestalProcessorData operator+(const PedestalProcessorData& lhs, const PedestalProcessorData& rhs);

/// @brief Output stream operator for PedestalProcessorData::ChannelIndexException
/// @param stream Stream used for printing
/// @param ex Exception to be printed
/// @return Stream after printing
std::ostream& operator<<(std::ostream& stream, const PedestalProcessorData::ChannelIndexException& ex);

} // namespace o2::emcal

#endif