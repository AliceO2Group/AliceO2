// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ALICEO2_EMCAL_CONSTANTS_H_
#define ALICEO2_EMCAL_CONSTANTS_H_

#include <Rtypes.h>
#include <iosfwd>
#include <exception>
#include <cstdint>

namespace o2
{
namespace emcal
{
enum {
  EMCAL_MODULES = 22,   ///< Number of modules, 12 for EMCal + 8 for DCAL
  EMCAL_ROWS = 24,      ///< Number of rows per module for EMCAL
  EMCAL_COLS = 48,      ///< Number of columns per module for EMCAL
  EMCAL_LEDREFS = 24,   ///< Number of LEDs (reference/monitors) per module for EMCAL; one per StripModule
  EMCAL_TEMPSENSORS = 8 ///< Number Temperature sensors per module for EMCAL
};

/// \enum ChannelType_t
/// \brief Type of a raw data channel
enum ChannelType_t {
  HIGH_GAIN, ///< High gain channel
  LOW_GAIN,  ///< Low gain channel
  TRU,       ///< TRU channel
  LEDMON     ///< LED monitor channel
};

/// \class InvalidChanneltypeException
/// \brief Error handling invalid channel types
class InvalidChanneltypeException : public std::exception
{
 public:
  /// \brief Constructor initializing the exception
  /// \param caloflag Calo flag in mapping responsible for exception
  InvalidChanneltypeException(int caloflag) : std::exception(),
                                              mCaloFlag(caloflag),
                                              mMessage("Wrong channel type value found (" + std::to_string(caloflag) + ")! Should be 0 ,1, 2 or 3 !")
  {
  }
  /// \brief Destructor
  ~InvalidChanneltypeException() noexcept final = default;

  /// \brief Access to error message of the exception
  /// \return Error messag
  const char* what() const noexcept final { return "Invalid caloflag, no channel type matching"; }

  /// \brief Access to calo flag responsible for the exception
  /// \return Calo flag
  int getCaloflag() const noexcept { return mCaloFlag; }

 private:
  int mCaloFlag;        ///< Calo flag responsible for exception
  std::string mMessage; ///< Error messagete:
};

/// \brief Create string representation of the channel type object
/// \param in chantype Channel type object
/// \return String representation of the channel type object
std::string channelTypeToString(ChannelType_t chantype);

/// \brief integer representation of the channel type object
/// \param chantype Channel type object
/// \return Integer representation of the channel type object
int channelTypeToInt(ChannelType_t chantype);

/// \brief Convert integer number to channel type object
/// \param chantype Number representation of the channel type
/// \return Channel type corresponding to number
/// \throw InvalidChanneltypeException in case the number doesn't match to a channel type
ChannelType_t intToChannelType(int chantype);

/// \brief Stream operator for ChannelType_t
/// \param stream Output stream where to put the channel type on
/// \param chantype ChannelType_t object to be put on the stream
/// \return Resulting stream
std::ostream& operator<<(std::ostream& stream, ChannelType_t chantype);

namespace constants
{

constexpr Double_t EMCAL_TIMESAMPLE = 100.;  ///< Width of a timebin in nanoseconds
constexpr Double_t EMCAL_ADCENERGY = 0.0167; ///< Energy of one ADC count in GeV/c^2
constexpr Int_t EMCAL_MAXTIMEBINS = 15;      ///< Maximum number of time bins for time response
} // namespace constants

} // namespace emcal
} // namespace o2
#endif
