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

/// \file GeometryBuilder.h
/// \brief Class describing MFT Geometry Builder
/// \author Raphael Tieulent <raphael.tieulent@cern.ch>
/// \date 09/06/2015

#ifndef ALICEO2_MFT_GEOMETRYBUILDER_H_
#define ALICEO2_MFT_GEOMETRYBUILDER_H_

namespace o2
{
namespace mft
{

class GeometryBuilder
{

 public:
  GeometryBuilder() = default;
  ~GeometryBuilder() = default;

  void buildGeometry();

 private:
  ClassDef(GeometryBuilder, 1);
};
} // namespace mft
} // namespace o2

#endif
