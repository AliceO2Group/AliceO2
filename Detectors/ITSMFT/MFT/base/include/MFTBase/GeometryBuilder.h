// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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

#include "TNamed.h"

namespace o2 {
namespace MFT {

class GeometryBuilder : public TNamed {

 public:

  GeometryBuilder();
  ~GeometryBuilder() override;

  void buildGeometry();

 private:

  ClassDefOverride(GeometryBuilder, 1)

};

}
}

#endif
