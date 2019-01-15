#include "ClEnv.h"

#include <gpucf/errors/FileErrors.h>
#include <gpucf/errors/CLErrors.h>
#include <gpucf/log.h>

#include <fstream>
#include <stdexcept>


using namespace gpucf;
namespace fs = filesystem;


ClEnv::ClEnv(const fs::path &srcDir, size_t gid) 
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
    
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) 
    {
        throw NoGpuFoundError();
    }
    
    context = cl::Context(devices, nullptr, nullptr, nullptr);

    std::string deviceName;
    getDevice().getInfo(CL_DEVICE_NAME, &deviceName);
    log::Info() << "Running on device " << deviceName;
}

cl::Program ClEnv::buildFromSrc(const fs::path &srcFile) 
{
    cl::Program::Sources src = loadSrc(srcFile);

    cl::Program prg(context, src);

    try 
    {
        std::string buildOpts = "-Werror -I" + sourceDir.str();
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

cl::Program::Sources ClEnv::loadSrc(const fs::path &srcFile) 
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

    cl::Program::Sources source({code});

    return source;
}
 
// vim: set ts=4 sw=4 sts=4 expandtab:
