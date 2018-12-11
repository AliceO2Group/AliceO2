// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file VertexerCPU.h
/// \brief
///

#ifndef VERTEXERCPU_H_
#define VERTEXERCPU_H_

// #include <array>
// #include <chrono>
// #include <cmath>
// #include <fstream>
// #include <iomanip>
// #include <iosfwd>
// #include <memory>
// #include <utility>
//
#include "ITStracking/VertexerBase.h"
// #include "ITStracking/Configuration.h"
// #include "ITStracking/Definitions.h"
// #include "ITStracking/MathUtils.h"
// #include "ITStracking/PrimaryVertexContext.h"
// #include "ITStracking/Road.h"

namespace o2
{
namespace ITS
{

class VertexerCPU : public VertexerBase
{
 public:
  // VertexerCPU() { }
  // virtual ~VertexerCPU() { }


  // void computeLayerTracklets() final;
  // void computeLayerCells() final;

 protected:

  // std::vector<std::vector<Tracklet>> mTracklets;
  // std::vector<std::vector<Cell>> mCells;
};

}
}

#endif /* VERTEXERCPU_H_ */
