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

/// \file MagFieldParam.h
/// \brief Definition of the MagFieldParam: ALICE mag. field type enumerations
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_FIELD_MAGFIELDPARAM_H_
#define ALICEO2_FIELD_MAGFIELDPARAM_H_

namespace o2
{
namespace field
{

/// Field map and beam type enumerations.
///
/// This used to be a FairParGenericSet container written into the FairRoot
/// runtime database. That path was never populated nor read back in production
/// (field settings come from CCDB/SimConfig via SimFieldUtils), so only the
/// enumerations - which are used throughout O2 - are kept.
struct MagFieldParam {
  enum BMap_t {
    k2kG,
    k5kG,
    k5kGUniform,
    kNFieldTypes
  };
  enum BeamType_t {
    kNoBeamField,
    kBeamTypepp,
    kBeamTypeAA,
    kBeamTypepA,
    kBeamTypeAp
  };
};

} // namespace field
} // namespace o2

#endif
