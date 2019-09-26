// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/common/Timestamp.h>

#include <nonstd/optional.h>

#include <CL/cl2.h>

namespace gpucf
{

class Event
{

 public:
  Event();

  cl::Event* get();

  Timestamp queued() const;
  Timestamp submitted() const;
  Timestamp start() const;
  Timestamp end() const;

 private:
  cl::Event event;

  Timestamp profilingInfo(cl_profiling_info) const;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
