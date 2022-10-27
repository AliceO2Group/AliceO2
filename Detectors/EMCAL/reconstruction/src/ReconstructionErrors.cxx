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

} // namespace reconstructionerrors

} // namespace emcal

} // namespace o2