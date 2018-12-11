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
/// \file VertexerCUDA.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_VERTEXERCUDA_H_
#define TRACKINGITSU_INCLUDE_VERTEXERCUDA_H_

#include "ITStracking/VertexerBase.h"

namespace o2
{
namespace ITS
{

class VertexerCUDA : public VertexerBase
{
 public:
  VertexerCUDA();
  virtual ~VertexerCUDA();
  void computeLayerTracklets() final;

};

extern "C" VertexerBase* createVertexerCUDA();
}
}

#endif /* TRACKINGITSU_INCLUDE_VERTEXERCUDA_H_ */
