// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file opencl_compiler_structs.h
/// \author David Rohr

struct _makefiles_opencl_platform_info {
  char platform_profile[64];
  char platform_version[64];
  char platform_name[64];
  char platform_vendor[64];
  cl_uint count;
};

struct _makefiles_opencl_device_info {
  char device_name[64];
  char device_vendor[64];
  cl_uint nbits;
  size_t binary_size;
};
