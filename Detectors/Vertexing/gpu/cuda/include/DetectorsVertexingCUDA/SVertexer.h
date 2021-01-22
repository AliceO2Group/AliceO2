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
/// \file SVertexer.h
/// \ author: mconcas@cern.ch

#include "GPUCommonDef.h"

namespace o2
{
namespace vertexing
{

void hello_util();

class SVertexerCUDA final
{
 public:
  SVertexerCUDA() = default;
  virtual ~SVertexerCUDA() = default;
  void hello();
};

// Steers
void SVertexerCUDA::hello()
{
  hello_util();
}
} // namespace vertexing
} // namespace o2