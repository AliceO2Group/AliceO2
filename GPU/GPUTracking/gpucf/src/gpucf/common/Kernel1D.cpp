// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Kernel1D.h"

#include <gpucf/common/log.h>

using namespace gpucf;

Kernel1D::Kernel1D(const std::string& name, cl::Program prg)
  : kernel(prg, name.c_str()), name(name)
{
}

void Kernel1D::call(
  size_t offset,
  size_t workitems,
  size_t local,
  cl::CommandQueue queue)
{
  ASSERT(workitems > 0);

  try {
    queue.enqueueNDRangeKernel(
      kernel,
      cl::NDRange(offset),
      cl::NDRange(workitems),
      cl::NDRange(local),
      nullptr,
      event.get());
  } catch (const cl::Error& err) {
    log::Error() << "Kernel " << name << " throws error "
                 << err.what() << "(" << log::clErrToStr(err.err()) << ")";

    throw err;
  }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
