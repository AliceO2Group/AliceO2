// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "log.h"

#include <CL/cl.h>

#include <unordered_map>

namespace gpucf
{
namespace log
{
#define ERR_STR_PAIR(err) \
  {                       \
    err, #err             \
  }

std::string clErrToStr(cl_int errcode)
{
  static const std::unordered_map<int, std::string> mapErrToStr =
    {
      ERR_STR_PAIR(CL_SUCCESS),
      ERR_STR_PAIR(CL_DEVICE_NOT_FOUND),
      ERR_STR_PAIR(CL_DEVICE_NOT_AVAILABLE),
      ERR_STR_PAIR(CL_COMPILER_NOT_AVAILABLE),
      ERR_STR_PAIR(CL_MEM_OBJECT_ALLOCATION_FAILURE),
      ERR_STR_PAIR(CL_OUT_OF_RESOURCES),
      ERR_STR_PAIR(CL_OUT_OF_HOST_MEMORY),
      ERR_STR_PAIR(CL_PROFILING_INFO_NOT_AVAILABLE),
      ERR_STR_PAIR(CL_MEM_COPY_OVERLAP),
      ERR_STR_PAIR(CL_IMAGE_FORMAT_MISMATCH),
      ERR_STR_PAIR(CL_IMAGE_FORMAT_NOT_SUPPORTED),
      ERR_STR_PAIR(CL_BUILD_PROGRAM_FAILURE),
      ERR_STR_PAIR(CL_MAP_FAILURE),
      ERR_STR_PAIR(CL_MISALIGNED_SUB_BUFFER_OFFSET),
      ERR_STR_PAIR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST),
      ERR_STR_PAIR(CL_COMPILE_PROGRAM_FAILURE),
      ERR_STR_PAIR(CL_LINKER_NOT_AVAILABLE),
      ERR_STR_PAIR(CL_LINK_PROGRAM_FAILURE),
      ERR_STR_PAIR(CL_DEVICE_PARTITION_FAILED),
      ERR_STR_PAIR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE),
      ERR_STR_PAIR(CL_INVALID_VALUE),
      ERR_STR_PAIR(CL_INVALID_DEVICE_TYPE),
      ERR_STR_PAIR(CL_INVALID_PLATFORM),
      ERR_STR_PAIR(CL_INVALID_DEVICE),
      ERR_STR_PAIR(CL_INVALID_CONTEXT),
      ERR_STR_PAIR(CL_INVALID_QUEUE_PROPERTIES),
      ERR_STR_PAIR(CL_INVALID_COMMAND_QUEUE),
      ERR_STR_PAIR(CL_INVALID_HOST_PTR),
      ERR_STR_PAIR(CL_INVALID_MEM_OBJECT),
      ERR_STR_PAIR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR),
      ERR_STR_PAIR(CL_INVALID_IMAGE_SIZE),
      ERR_STR_PAIR(CL_INVALID_SAMPLER),
      ERR_STR_PAIR(CL_INVALID_BINARY),
      ERR_STR_PAIR(CL_INVALID_BUILD_OPTIONS),
      ERR_STR_PAIR(CL_INVALID_PROGRAM),
      ERR_STR_PAIR(CL_INVALID_PROGRAM_EXECUTABLE),
      ERR_STR_PAIR(CL_INVALID_KERNEL_NAME),
      ERR_STR_PAIR(CL_INVALID_KERNEL_DEFINITION),
      ERR_STR_PAIR(CL_INVALID_KERNEL),
      ERR_STR_PAIR(CL_INVALID_ARG_INDEX),
      ERR_STR_PAIR(CL_INVALID_ARG_VALUE),
      ERR_STR_PAIR(CL_INVALID_ARG_SIZE),
      ERR_STR_PAIR(CL_INVALID_KERNEL_ARGS),
      ERR_STR_PAIR(CL_INVALID_WORK_DIMENSION),
      ERR_STR_PAIR(CL_INVALID_WORK_GROUP_SIZE),
      ERR_STR_PAIR(CL_INVALID_WORK_ITEM_SIZE),
      ERR_STR_PAIR(CL_INVALID_GLOBAL_OFFSET),
      ERR_STR_PAIR(CL_INVALID_EVENT_WAIT_LIST),
      ERR_STR_PAIR(CL_INVALID_EVENT),
      ERR_STR_PAIR(CL_INVALID_OPERATION),
      ERR_STR_PAIR(CL_INVALID_GL_OBJECT),
      ERR_STR_PAIR(CL_INVALID_BUFFER_SIZE),
      ERR_STR_PAIR(CL_INVALID_MIP_LEVEL),
      ERR_STR_PAIR(CL_INVALID_GLOBAL_WORK_SIZE),
      ERR_STR_PAIR(CL_INVALID_PROPERTY),
      ERR_STR_PAIR(CL_INVALID_IMAGE_DESCRIPTOR),
      ERR_STR_PAIR(CL_INVALID_COMPILER_OPTIONS),
      ERR_STR_PAIR(CL_INVALID_LINKER_OPTIONS),
      ERR_STR_PAIR(CL_INVALID_DEVICE_PARTITION_COUNT),
      ERR_STR_PAIR(CL_INVALID_PIPE_SIZE),
      ERR_STR_PAIR(CL_INVALID_DEVICE_QUEUE),
    };

  auto got = mapErrToStr.find(errcode);

  if (got == mapErrToStr.end()) {
    return "UNKNOWN_ERROR";
  }

  return got->second;
}

const char* levelToStr(Level lvl)
{
  switch (lvl) {
    case Level::Debug:
      return "[Debug]";
    case Level::Info:
      return "[Info ]";
    case Level::Error:
      return "[Error]";
  }
  return "";
}

std::ostream& operator<<(std::ostream& o, Level lvl)
{
  return o << levelToStr(lvl);
}

} // namespace log
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
