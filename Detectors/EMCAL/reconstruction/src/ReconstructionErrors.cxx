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
#include <cassert>
#include "EMCALReconstruction/ReconstructionErrors.h"

namespace o2
{
namespace emcal
{

namespace reconstructionerrors
{

GeometryError_t getGeometryErrorFromErrorCode(unsigned int errorcode)
{
  assert(errorcode < getNumberOfGeometryErrorCodes());
  switch (errorcode) {
    case 0:
      return GeometryError_t::CELL_RANGE_EXCEED;
    case 1:
      return GeometryError_t::CELL_INDEX_NEGATIVE;
    default:
      return GeometryError_t::UNKNOWN_ERROR;
  }
}

const char* getGeometryErrorName(GeometryError_t errortype)
{
  switch (errortype) {
    case GeometryError_t::CELL_RANGE_EXCEED:
      return "CellRangeExceed";
    case GeometryError_t::CELL_INDEX_NEGATIVE:
      return "CellIndexNegative";
    default:
      return "UnknownError";
  }
}

const char* getGeometryErrorName(unsigned int errorcode)
{
  return getGeometryErrorName(getGeometryErrorFromErrorCode(errorcode));
}

const char* getGeometryErrorTitle(GeometryError_t errortype)
{
  switch (errortype) {
    case GeometryError_t::CELL_RANGE_EXCEED:
      return "Cell ID outside range";
    case GeometryError_t::CELL_INDEX_NEGATIVE:
      return "Cell ID corrupted";
    default:
      return "UnknownError";
  };
}

const char* getGeometryErrorTitle(unsigned int errorcode)
{
  return getGeometryErrorTitle(getGeometryErrorFromErrorCode(errorcode));
}

const char* getGeometryErrorDescription(GeometryError_t errortype)
{
  switch (errortype) {
    case GeometryError_t::CELL_RANGE_EXCEED:
      return "Cell index exceeding valid rage";
    case GeometryError_t::CELL_INDEX_NEGATIVE:
      return "Cell index corrupted (i.e. negative)";
    default:
      return "UnknownError";
  };
}

const char* getGeometryErrorDescription(unsigned int errorcode)
{
  return getGeometryErrorDescription(getGeometryErrorFromErrorCode(errorcode));
}

GainError_t getGainErrorFromErrorCode(unsigned int errorcode)
{
  assert(errorcode < getNumberOfGainErrorCodes());
  switch (errorcode) {
    case 0:
      return GainError_t::LGNOHG;
    case 1:
      return GainError_t::HGNOLG;
    default:
      return GainError_t::UNKNOWN_ERROR;
  };
}

const char* getGainErrorName(GainError_t errortype)
{
  switch (errortype) {
    case GainError_t::LGNOHG:
      return "HGnoLG";
    case GainError_t::HGNOLG:
      return "LGnoHG";
    default:
      return "UnknownError";
  };
}

const char* getGainErrorName(unsigned int errorcode)
{
  return getGainErrorName(getGainErrorFromErrorCode(errorcode));
}

const char* getGainErrorTitle(GainError_t errortype)
{
  switch (errortype) {
    case GainError_t::LGNOHG:
      return "High Gain missing";
    case GainError_t::HGNOLG:
      return "Low Gain missing";
    default:
      return "UnknownError";
  };
}

const char* getGainErrorTitle(unsigned int errorcode)
{
  return getGainErrorTitle(getGainErrorFromErrorCode(errorcode));
}

const char* getGainErrorDescription(GainError_t errortype)
{
  switch (errortype) {
    case GainError_t::LGNOHG:
      return "HG missing for LG below HGLG transition";
    case GainError_t::HGNOLG:
      return "LG not found for saturated HG";
    default:
      return "UnknownError";
  };
}

const char* getGainErrorDescription(unsigned int errorcode)
{
  return getGainErrorDescription(getGainErrorFromErrorCode(errorcode));
}

TRUDecodingError_t getTRUDecodingErrorFromErrorCode(unsigned int errorcode)
{
  switch (errorcode) {
    case 0:
      return TRUDecodingError_t::TRU_INDEX_INVALID;
    case 1:
      return TRUDecodingError_t::PATCH_INDEX_INVALID;
    case 2:
      return TRUDecodingError_t::FASTOR_INDEX_INVALID;
    default:
      return TRUDecodingError_t::UNKNOWN_ERROR;
  }
}

const char* getTRUDecodingErrorName(TRUDecodingError_t errortype)
{
  switch (errortype) {
    case TRUDecodingError_t::TRU_INDEX_INVALID:
      return "TRUIndexInvalid";
    case TRUDecodingError_t::PATCH_INDEX_INVALID:
      return "PatchIndexInvalid";
    case TRUDecodingError_t::FASTOR_INDEX_INVALID:
      return "FastORIndexInvalid";
    default:
      return "UnknownError";
  }
}

const char* getTRUDecodingErrorName(unsigned int errorcode)
{
  return getTRUDecodingErrorName(getTRUDecodingErrorFromErrorCode(errorcode));
}

const char* getTRUDecodingErrorTitle(TRUDecodingError_t errortype)
{
  switch (errortype) {
    case TRUDecodingError_t::TRU_INDEX_INVALID:
      return "TRU index invalid";
    case TRUDecodingError_t::PATCH_INDEX_INVALID:
      return "Patch index invalid";
    case TRUDecodingError_t::FASTOR_INDEX_INVALID:
      return "FastOR index invalid";
    default:
      return "Unknown error";
  }
}

const char* getTRUDecodingErrorTitle(unsigned int errortype)
{
  return getTRUDecodingErrorTitle(getTRUDecodingErrorFromErrorCode(errortype));
}

const char* getTRUDecodingErrorErrorDescription(TRUDecodingError_t errortype)
{
  switch (errortype) {
    case TRUDecodingError_t::TRU_INDEX_INVALID:
      return "TRU index is invalid";
    case TRUDecodingError_t::PATCH_INDEX_INVALID:
      return "Patch index is invalid";
    case TRUDecodingError_t::FASTOR_INDEX_INVALID:
      return "FastOR index is invalid";
    default:
      return "Unknown error";
  }
}

const char* getTRUDecodingErrorErrorDescription(unsigned int errorcode)
{
  return getTRUDecodingErrorErrorDescription(getTRUDecodingErrorFromErrorCode(errorcode));
}

} // namespace reconstructionerrors

} // namespace emcal

} // namespace o2