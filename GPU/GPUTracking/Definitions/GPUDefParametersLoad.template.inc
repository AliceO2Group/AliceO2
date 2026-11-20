// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUDefParametersLoad.inc
/// \author David Rohr

#include "GPUDefParametersLoadPrepare.h"

#define GPUCA_M_LB_EMPTY_(...) 0
#define GPUCA_M_LB_EMPTY_1(...) __VA_ARGS__
#define GPUCA_M_LB_EMPTY0(...) GPUCA_M_CAT(GPUCA_M_LB_EMPTY_, __VA_OPT__(1))(__VA_ARGS__)

#include "GPUDefParametersRuntime.h"
#include "GPUDefMacros.h"
#include <sstream>
#include <string>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace o2::gpu::internal
{

static GPUDefParameters GPUDefParametersLoad()
{
  return GPUDefParameters{
    // clang-format off
    {$<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,GPUCA_M_LB_EMPTY0(GPUCA_M_FIRST(GPUCA_LB_>,APPEND,))>,$<COMMA>>},
    {$<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,GPUCA_M_FIRST(GPUCA_M_SHIFT(GPUCA_LB_>,APPEND,$<COMMA>0))>,$<COMMA>>},
    {$<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,GPUCA_M_FIRST(GPUCA_M_SHIFT(GPUCA_M_SHIFT(GPUCA_LB_>,APPEND,$<COMMA>0$<COMMA>0)))>,$<COMMA>>},
    $<JOIN:$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_PARAMS>,PREPEND,GPUCA_PAR_>,$<COMMA>>,
    $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_STRING_PARAMS>,PREPEND,GPUCA_M_STR(GPUCA_PAR_>,APPEND,)>,$<COMMA>>
    // clang-format on
  };
}

#define GPUCA_EXPORT_KERNEL_LB(name)                                                                                                                     \
  if (par.par_LB_maxThreads[i] > 0) {                                                                                                                    \
    o << "#define GPUCA_LB_" GPUCA_M_STR(name) " " << par.par_LB_maxThreads[i];                                                                          \
    if (par.par_LB_minBlocks[i] > 0) {                                                                                                                   \
      o << ", " << GPUCA_CEIL_INT_DIV(par.par_LB_maxThreads[i] * par.par_LB_minBlocks[i], (minBlockFactor ? minBlockFactor : par.par_LB_maxThreads[i])); \
    }                                                                                                                                                    \
    if (!forRTC && par.par_LB_forceBlocks[i] > 0) {                                                                                                      \
      o << ", " << par.par_LB_forceBlocks[i];                                                                                                            \
    }                                                                                                                                                    \
    o << "\n";                                                                                                                                           \
  }                                                                                                                                                      \
  i++;

#define GPUCA_EXPORT_KERNEL_PARAM(name) \
  o << "#define GPUCA_PAR_" GPUCA_M_STR(name) " " << GPUCA_M_CAT(par.par_, name) << "\n";

static std::string GPUDefParametersExport(const GPUDefParameters& par, bool forRTC, int32_t minBlockFactor = 0)
{
  std::stringstream o; // clang-format off
  int32_t i = 0;
  $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,GPUCA_EXPORT_KERNEL_LB(>,APPEND,)>,
  >
  $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_PARAMS>,PREPEND,GPUCA_EXPORT_KERNEL_PARAM(>,APPEND,)>,
  >
  $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_STRING_PARAMS>,PREPEND,GPUCA_EXPORT_KERNEL_PARAM(>,APPEND,)>,
  >
  return o.str(); // clang-format on
}

#define GPUCA_OVERRIDE_KERNEL_LB(name)                                                                                 \
  if (key == GPUCA_M_STR(name)) {                                                                                      \
    par.par_LB_maxThreads[i] = par.par_LB_minBlocks[i] = par.par_LB_forceBlocks[i] = 0;                                \
    std::stringstream v(val.front() == '[' && val.back() == ']' ? val.substr(1, val.size() - 2) : val);                \
    std::string tok;                                                                                                   \
    for (int32_t j = 0; std::getline(v, tok, ','); j++) {                                                              \
      if (j >= 3) {                                                                                                    \
        throw std::runtime_error("Too many values in launch bounds override " + entry);                                \
      }                                                                                                                \
      (j == 0 ? par.par_LB_maxThreads : (j == 1 ? par.par_LB_minBlocks : par.par_LB_forceBlocks))[i] = std::stoi(tok); \
    }                                                                                                                  \
    found = true;                                                                                                      \
  }                                                                                                                    \
  i++;

#define GPUCA_OVERRIDE_KERNEL_PARAM(name)         \
  if (key == GPUCA_M_STR(name)) {                 \
    GPUCA_M_CAT(par.par_, name) = std::stoi(val); \
    found = true;                                 \
  }

#define GPUCA_OVERRIDE_KERNEL_STRING_PARAM(name)                                 \
  if (key == GPUCA_M_STR(name)) {                                                \
    if (val.size() >= sizeof(GPUCA_M_CAT(par.par_, name))) {                     \
      throw std::runtime_error("Value too long in parameter override " + entry); \
    }                                                                            \
    strcpy(GPUCA_M_CAT(par.par_, name), val.c_str());                            \
    found = true;                                                                \
  }

static void GPUDefParametersOverride(GPUDefParameters& par, const std::vector<std::string>& overrides) // Entries name=value: name is a kernel name as in the parameter file for launch bounds (value n or [maxThreads, minBlocks, forceBlocks], values not given are reset), or the name of a parameter
{
  const auto trim = [](const std::string& str) {
    const size_t first = str.find_first_not_of(' '), last = str.find_last_not_of(' ');
    return first == std::string::npos ? std::string() : str.substr(first, last - first + 1);
  };
  for (const std::string& entry : overrides) {
    const size_t eq = entry.find('=');
    const std::string key = trim(entry.substr(0, eq)), val = eq == std::string::npos ? std::string() : trim(entry.substr(eq + 1));
    if (key.empty() || val.empty()) {
      throw std::runtime_error("Invalid parameter override " + entry);
    }
    bool found = false;
    int32_t i = 0;
    try { // clang-format off
      $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,GPUCA_OVERRIDE_KERNEL_LB(>,APPEND,)>,
      >
      $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_PARAMS>,PREPEND,GPUCA_OVERRIDE_KERNEL_PARAM(>,APPEND,)>,
      >
      $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_STRING_PARAMS>,PREPEND,GPUCA_OVERRIDE_KERNEL_STRING_PARAM(>,APPEND,)>,
      >
    } catch (const std::logic_error&) { // clang-format on
      throw std::runtime_error("Invalid value in parameter override " + entry);
    }
    if (!found) {
      throw std::runtime_error("Unknown parameter in override " + entry);
    }
  }
}

} // namespace o2::gpu::internal
