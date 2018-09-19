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
/// \file Stream.cu
/// \brief
///

#include "ITStrackingCUDA/Stream.h"

#include <cuda_runtime.h>

namespace o2
{
namespace ITS
{
namespace GPU
{

Stream::Stream()
{
  cudaStreamCreateWithFlags(&mStream, cudaStreamNonBlocking);
}

Stream::~Stream()
{
  cudaStreamDestroy(mStream);
}

const GPUStream& Stream::get() const
{
  return mStream;
}

}
}
}
