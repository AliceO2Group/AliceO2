// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#if !defined(SHARED_TYPES_H)
#    define  SHARED_TYPES_H

#if defined(__OPENCL_VERSION__)
#  define IS_CL_DEVICE 1
#else
#  define IS_CL_DEVICE 0
#endif

#define IS_CL_HOST (!IS_CL_DEVICE)


#if IS_CL_DEVICE
#  define CL_PREFIX(type) type
#else
#  include <CL/cl_platform.h>
#  define CL_PREFIX(type) cl_ ## type
#endif

#define SHARED_INT     CL_PREFIX(int)
#define SHARED_UINT    CL_PREFIX(uint)
#define SHARED_FLOAT   CL_PREFIX(float)
#define SHARED_UCHAR   CL_PREFIX(uchar)
#define SHARED_HALF    CL_PREFIX(half)
#define SHARED_USHORT  CL_PREFIX(ushort)

#endif //!defined(SHARED_TYPES_H)

// vim: set ts=4 sw=4 sts=4 expandtab:
