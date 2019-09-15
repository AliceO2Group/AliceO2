// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "ClEnv.h"

#include <gpucf/common/log.h>
#include <gpucf/errors/FileErrors.h>
#include <gpucf/errors/CLErrors.h>

#include <fstream>
#include <stdexcept>


using namespace gpucf;
namespace fs = filesystem;


const std::vector<fs::path> ClEnv::srcs = {
    "clusterFinder.cl",
    "streamCompaction.cl",
};


ClEnv::ClEnv(const fs::path &srcDir, ClusterFinderConfig cfg, size_t gid, bool useCpu)
    : gpuId(gid) 
    , sourceDir(srcDir) 
{
    if (!sourceDir.is_directory()) 
    {
        throw DirectoryNotFoundError(sourceDir);
    }

    
    cl::Platform::get(&platforms); 
    if (platforms.empty())
    {
        throw NoPlatformFoundError(); 
    }
    
    cl_device_type dtype = (useCpu) ? CL_DEVICE_TYPE_CPU 
                                    : CL_DEVICE_TYPE_GPU;
    platforms.front().getDevices(dtype, &devices);

    if (devices.empty()) 
    {
        throw NoGpuFoundError();
    }
    
    context = cl::Context(devices, nullptr, nullptr, nullptr);

    std::string deviceName;
    getDevice().getInfo(CL_DEVICE_NAME, &deviceName);
    log::Info() << "Running on device " << deviceName;


    switch (cfg.layout)
    {
    #define MEMORY_LAYOUT(name, def, desc) \
        case ChargemapLayout::name: \
            addDefine(def); \
            break;
    #include <gpucf/algorithms/ClusterFinderFlags.def>
    }

    switch (cfg.clusterbuilder)
    {
    #define CLUSTER_BUILDER(name, def, desc) \
        case ClusterBuilder::name: \
            addDefine(def); \
            break;
    #include <gpucf/algorithms/ClusterFinderFlags.def>
    }

    #define CLUSTER_FINDER_FLAG(name, val, def, desc) \
        if (cfg.name) \
        { \
            addDefine(def); \
        }
    #include <gpucf/algorithms/ClusterFinderFlags.def>

    #define CLUSTER_FINDER_PARAM(name, val, def, desc) \
        addDefine(std::string(def) + std::string("=") \
                + std::to_string(cfg.name));
    #include <gpucf/algorithms/ClusterFinderFlags.def>

#if defined(NDEBUG)
    addDefine("NDEBUG");
#endif

    program = buildFromSrc(useCpu);
}

cl::Program ClEnv::buildFromSrc(bool cpudebug)
{
    cl::Program::Sources src = loadSrc(srcs);

    cl::Program prg(context, src);

    std::string buildOpts = "-Werror";
    buildOpts += " -I" + sourceDir.str();
    buildOpts += " -cl-std=CL2.0";

    if (cpudebug)
    {
        buildOpts += " -g -O0"; 
    }

    for (const std::string &def : defines)
    {
        buildOpts += " -D" + def;
    }

    log::Info() << "Build flags: " << buildOpts;

    try 
    {
        prg.build(devices, buildOpts.c_str());
    } 
    catch (const cl::BuildError &) 
    {
        auto buildInfo = prg.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
        for (auto &pair : buildInfo) 
        {
            std::cerr << pair.second << std::endl << std::endl;
        }
        throw BuildFailedError();
    }

    return prg;
}

cl::Program::Sources ClEnv::loadSrc(const std::vector<fs::path> &srcFiles) 
{
    std::vector<std::string> codes;

    for (const fs::path &srcFile : srcFiles)
    {

        fs::path file = sourceDir / srcFile; 

        log::Info() << "Opening cl-source " << file;

        if (!file.exists()) 
        {
            throw FileNotFoundError(file);
        }

        std::ifstream src(file.str());
        std::string code( (std::istreambuf_iterator<char>(src)),
                std::istreambuf_iterator<char>());

        codes.push_back(code);
    }

    cl::Program::Sources source(codes);

    return source;
}

void ClEnv::addDefine(const std::string &def)
{
    defines.push_back(def);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
