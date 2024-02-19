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
  LOW_GAIN,  ///< Low gain channel
  HIGH_GAIN, ///< High gain channel
  TRU,       ///< TRU channel
  LEDMON     ///< LED monitor channel
};

/// \class InvalidChanneltypeException
/// \brief Error handling invalid channel types
class InvalidChanneltypeException final : public std::exception
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

constexpr int OVERFLOWCUT = 950;               ///< sample overflow
constexpr int LG_SUPPRESSION_CUT = 880;        ///< LG bunch suppression ADC value
constexpr int ORDER = 2;                       ///< Order of shaping stages of the signal conditioning unit
constexpr double TAU = 2.35;                   ///< Approximate shaping time
constexpr Double_t EMCAL_TIMESAMPLE = 100.;    ///< Width of a timebin in nanoseconds
constexpr Double_t EMCAL_ADCENERGY = 0.0162;   ///< Energy of one ADC count in GeV/c^2
constexpr Int_t EMCAL_HGLGFACTOR = 16;         ///< Conversion from High to Low Gain
constexpr Int_t EMCAL_HGLGTRANSITION = 1024;   ///< Transition from High to Low Gain
constexpr Int_t EMCAL_MAXTIMEBINS = 15;        ///< Maximum number of time bins for time response
constexpr int MAX_RANGE_ADC = 0x3FF;           ///< Dynamic range of the ADCs (10 bit ADC)
constexpr double EMCAL_TRU_ADCENERGY = 0.0786; ///< resolution of the TRU digitizer, @TODO check exact value
} // namespace constants

namespace triggerbits
{
constexpr uint32_t Inc = 0x1 << 20; ///< trigger bit marking incomplete event
}

enum FitAlgorithm {
  Standard = 0,  ///< Standard raw fitter
  Gamma2 = 1,    ///< Gamma2 raw fitter
  NeuralNet = 2, ///< Neural net raw fitter
  NONE = 3
};

enum STUtype_t {
  ESTU = 0, ///< EMCAL STU
  DSTU = 1  ///< DCAL STU
};

namespace STUparam //[0]->EMCAL STU, [1]->DCAL STU
{
constexpr int FeeID[2] = {44, 45};                ///< FEE_ID in RDH
constexpr int NTRU[2] = {32, 14};                 ///< number of TRUs
constexpr int CFG_nWords[2] = {17, 17};           ///< number of configuration words
constexpr int L1JetIndex_nWords[2] = {11, 11};    ///< number of words with Jet indices
constexpr int L0index_nWords[2] = {96, 42};       ///< number of words with null data
constexpr int L1GammaIndex_nWords[2] = {128, 56}; ///< number of words with Gamma indices
constexpr int Raw_nWords[2] = {1536, 672};        ///< number of words with ADC
constexpr int SubregionsEta[2] = {12, 12};        ///< number of subregions over eta
constexpr int SubregionsPhi[2] = {16, 10};        ///< number of subregions over phi
constexpr int PaloadSizeFull[2] = {1928, 866};    ///< number of words in full payload = 1944/882-16
constexpr int PaloadSizeShort[2] = {391, 193};    ///< number of words in shorts payload = 407/209-16
} // namespace STUparam

namespace TRUparam
{
constexpr int Nchannels = 96;             ///< number of FastORs per TRU
constexpr int NchannelsOverEta = 8;       ///< number of FastORs over Eta for full- and 2/3-size SMs
constexpr int NchannelsOverPhi = 12;      ///< number of FastORs over Phi for full- and 2/3-size SMs
constexpr int NchannelsOverEta_long = 24; ///< number of FastORs over Eta for 1/3-size SMs
constexpr int NchannelsOverPhi_long = 4;  ///< number of FastORs over Phi for 1/3-size SMs
} // namespace TRUparam

} // namespace emcal
} // namespace o2
#endif
