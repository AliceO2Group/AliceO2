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
#ifndef ALICEO2_EMCAL_RECONSTRUCTIONERRORS_H
#define ALICEO2_EMCAL_RECONSTRUCTIONERRORS_H

namespace o2
{

namespace emcal
{

namespace reconstructionerrors
{

/// \class GeometryError_t
/// \brief Errors appearing in geometry access obtaining the tower ID
/// \ingroup EMCALreconstruction
///
/// Errors can appear during geometry access, either because the
/// cell number provided by the geometry is negative or because
/// the cell number exceeds the range of allowed cell indices
enum class GeometryError_t {
  CELL_RANGE_EXCEED,   ///< Requested cell value exceeding allowed cell range
  CELL_INDEX_NEGATIVE, ///< Requested cell value negative
  UNKNOWN_ERROR        ///< Unknown error code
};

/// \brief Get the number of geometry error codes
/// \return Number of geometry error codes
constexpr int getNumberOfGeometryErrorCodes() { return 2; }

/// \brief Convert geometry error type into numberic representation
/// \param errortype Geometry error type
/// \return Error code connected to error type
constexpr int getErrorCodeFromGeometryError(GeometryError_t errortype)
{
  switch (errortype) {
    case GeometryError_t::CELL_RANGE_EXCEED:
      return 0;
    case GeometryError_t::CELL_INDEX_NEGATIVE:
      return 1;
    default:
      return -1;
  }
}

/// \brief Convert error code to geometry error type
///
/// Attention: Error code must be a valid error code, handled
/// internally via assert.
///
/// \param errorcode Error code to be converted
/// \return Error type connected to error code
GeometryError_t getGeometryErrorFromErrorCode(unsigned int errorcode);

/// \brief Get name of a given geometry error type
///
/// Name is a short single word descriptor used i.e. in
/// object names.
///
/// \param errortype Error type of the geometry error
/// \return Name connected to geometry error type
const char* getGeometryErrorName(GeometryError_t errortype);

/// \brief Get name of a given geometry error code
///
/// Name is a short single word descriptor used i.e. in
/// object names. Attention: Error code must be a valid
/// geomentry error code.
///
/// \param errorcode Error code of the geometry error
/// \return Name connected to geometry error type
const char* getGeometryErrorName(unsigned int errorcode);

/// \brief Get title of a given geometry error type
///
/// Title is a short descriptor used i.e. in
/// histogram titles.
///
/// \param errortype Error type of the geometry error
/// \return Title connected to geometry error type
const char* getGeometryErrorTitle(GeometryError_t errortype);

/// \brief Get title of a given geometry error type
///
/// Title is a short descriptor used i.e. in
/// histogram titles. Attention: Error code must
/// be a valid geomentry error code.
///
/// \param errorcode Error code of the geometry error
/// \return Title connected to geometry error type
const char* getGeometryErrorTitle(unsigned int errorcode);

/// \brief Get detailed description of a given geometry error type
///
/// Provides a long description to be used i.e. in error messages.
///
/// \param errortype Error type of the geometry error
/// \return Detaied description connected to geometry error type
const char* getGeometryErrorDescription(GeometryError_t errortype);

/// \brief Get detailed description of a given geometry error type
///
/// Provides a long description to be used i.e. in error messages.
/// Attention: Error code must be a valid geomentry error code.
///
/// \param errortype Error type of the geometry error
/// \return Detaied description connected to geometry error type
const char* getGeometryErrorDescription(unsigned int errorcode);

/// \enum GainError_t
/// \brief Errors appearing when merging gain types
/// \ingroup EMCALreconstruction
///
/// Errors can appear when an expected gain type is missing,
/// either because the HG is saturated and the LG is missing
/// or because the LG is found for an ADC below HG/LG transition
/// and the HG is missing.
enum class GainError_t {
  LGNOHG,       ///< LG found below HG/LG transition, HG missing
  HGNOLG,       ///< HG saturated, LG missing
  UNKNOWN_ERROR ///< Unknown error code
};

/// \brief Get the number of gain error codes
/// \return Number of gain error codes
constexpr int getNumberOfGainErrorCodes() { return 2; }

/// \brief Convert gain error type into numberic representation
/// \param errortype Gain error type
/// \return Error code connected to error type
constexpr int getErrorCodeFromGainError(GainError_t errortype)
{
  switch (errortype) {
    case GainError_t::LGNOHG:
      return 0;
    case GainError_t::HGNOLG:
      return 1;
    default:
      return -1;
  };
}

/// \brief Convert error code to gain error type
///
/// Attention: Error code must be a valid error code, handled
/// internally via assert.
///
/// \param errorcode Error code to be converted
/// \return Error type connected to error code
GainError_t getGainErrorFromErrorCode(unsigned int errorcode);

/// \brief Get name of a given gain error type
///
/// Name is a short single word descriptor used i.e. in
/// object names.
///
/// \param errortype Error type of the gain type error
/// \return Name connected to gain type error
const char* getGainErrorName(GainError_t errortype);

/// \brief Get name of a given gain error code
///
/// Name is a short single word descriptor used i.e. in
/// object names. Attention: Error code must be a valid
/// gain type error code.
///
/// \param errorcode Error code of the gain type error
/// \return Name connected to gain type error
const char* getGainErrorName(unsigned int errorcode);

/// \brief Get title of a given gain error type
///
/// Title is a short descriptor used i.e. in
/// histogram titles.
///
/// \param errortype Error type of the gain type error
/// \return Title connected to gain type error
const char* getGainErrorTitle(GainError_t errortype);

/// \brief Get title of a given gain error type
///
/// Title is a short descriptor used i.e. in
/// histogram titles. Attention: Error code must
/// be a valid gain type error code.
///
/// \param errorcode Error code of the gain type error
/// \return Title connected to gain type error
const char* getGainErrorTitle(unsigned int errorcode);

/// \brief Get detailed description of a given gain error type
///
/// Provides a long description to be used i.e. in error messages.
///
/// \param errortype Error type of the gain type error
/// \return Detaied description connected to gain type error
const char* getGainErrorDescription(GainError_t errortype);

/// \brief Get detailed description of a given gain error type
///
/// Provides a long description to be used i.e. in error messages.
/// Attention: Error code must be a valid gain type error code.
///
/// \param errortype Error type of the gain type error
/// \return Detaied description connected to gain type error
const char* getGainErrorDescription(unsigned int errorcode);

/// \enum TRUDecodingError_t
/// \brief Errors related to TRU data decoding
/// \ingroup EMCALReconstruction
///
/// TRU decoding errors appear during the decoding of the TRU data.
/// They can be raised by the trigger mapping, in case an invalid
/// FastOR index or TRU index is called, or by the TRU data handler if
/// the patch index is outside range.
enum class TRUDecodingError_t {
  TRU_INDEX_INVALID,    ///< TRU index invalid
  PATCH_INDEX_INVALID,  ///< Patch index outside range
  FASTOR_INDEX_INVALID, ///< FastOR index unknown
  UNKNOWN_ERROR         ///< Unknown error type
};

/// \brief Get the number of TRU error codes
/// \return Number of TRU error codes
constexpr int getNumberOfTRUErrorCodes() { return 3; }

/// \brief Convert error code to TRU decoding error type
///
/// Attention: Error code must be a valid error code, handled
/// internally via assert.
///
/// \param errorcode Error code to be converted
/// \return Error type connected to error code
TRUDecodingError_t getTRUDecodingErrorFromErrorCode(unsigned int errorcode);

/// \brief Convert TRU decoding error type into numberic representation
/// \param errortype TRU decoding error type
/// \return Error code connected to error type
constexpr int getErrorCodeFromTRUDecodingError(TRUDecodingError_t error)
{
  switch (error) {
    case TRUDecodingError_t::TRU_INDEX_INVALID:
      return 0;
    case TRUDecodingError_t::PATCH_INDEX_INVALID:
      return 1;
    case TRUDecodingError_t::FASTOR_INDEX_INVALID:
      return 2;
    case TRUDecodingError_t::UNKNOWN_ERROR:
      return -1;
  }
}

/// \brief Get name of a given TRU decoding error type
///
/// Name is a short single word descriptor used i.e. in
/// object names.
///
/// \param errortype Error type of the TRU decoding error
/// \return Name connected to TRU decoding error type
const char* getTRUDecodingErrorName(TRUDecodingError_t errortype);

/// \brief Get name of a given TRU decoding error code
///
/// Name is a short single word descriptor used i.e. in
/// object names. Attention: Error code must be a valid
/// TRU decoding error code.
///
/// \param errorcode Error code of the TRU decoding error
/// \return Name connected to TRU decoding error type
const char* getTRUDecodingErrorName(unsigned int errorcode);

/// \brief Get title of a given TRU decoding error type
///
/// Title is a short descriptor used i.e. in
/// histogram titles.
///
/// \param errortype Error type of the TRU decoding error
/// \return Title connected to TRU decoding error
const char* getTRUDecodingErrorTitle(TRUDecodingError_t errortype);

/// \brief Get title of a given TRU decoding error type
///
/// Title is a short descriptor used i.e. in
/// histogram titles. Attention: Error code must
/// be a valid TRU decoding error code.
///
/// \param errorcode Error code of the TRU decoding error
/// \return Title connected to TRU decoding error
const char* getTRUDecodingErrorTitle(unsigned int errorcode);

/// \brief Get detailed description of a given TRU decoding error type
///
/// Provides a long description to be used i.e. in error messages.
///
/// \param errortype Error type of the TRU decoding error
/// \return Detaied description connected to TRU decoding error
const char* getTRUDecodingErrorErrorDescription(TRUDecodingError_t errortype);

/// \brief Get detailed description of a given TRU decoding error type
///
/// Provides a long description to be used i.e. in error messages.
/// Attention: Error code must be a valid TRU decoding error code.
///
/// \param errortype Error type of the TRU decoding error
/// \return Detaied description connected to TRU decoding error
const char* getTRUDecodingErrorErrorDescription(unsigned int errorcode);

} // namespace reconstructionerrors

} // namespace emcal

} // namespace o2

#endif // !ALICEO2_EMCAL_RECONSTRUCTIONERRORS_H