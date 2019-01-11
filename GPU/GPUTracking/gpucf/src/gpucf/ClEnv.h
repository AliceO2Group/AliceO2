#pragma once

#include <gpucf/cl.h>

#include <args/args.hxx>
#include <filesystem/path.h>

#include <memory>
#include <string>
#include <vector>


class ClEnv 
{

public:
    class Flags 
    {

    public:
        args::ValueFlag<std::string> clSrcDir;    
        args::ValueFlag<size_t> gpuId;

        Flags(args::Group &required, args::Group &optional)
            : clSrcDir(required, "clsrc", "Base directory of cl source files.",
                    {'s', "src"}) 
            , gpuId(optional, "gpuid", "Id of the gpu device.", 
                    {'g', "gpu"}, 0)
        {
        }

    };


    ClEnv(const filesystem::path &srcDir, size_t gpuId=0);
    ClEnv(Flags &flags) 
        : ClEnv(args::get(flags.clSrcDir), args::get(flags.gpuId)) 
    {
    }

    cl::Context getContext() const 
    { 
        return context; 
    }

    cl::Device  getDevice()  const 
    { 
        return devices[gpuId]; 
    }

    cl::Program          buildFromSrc(const filesystem::path &srcFile);
    cl::Program::Sources loadSrc(const filesystem::path &srcFile);
    
private:
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    size_t gpuId;

    cl::Context context;

    filesystem::path sourceDir;
};

// vim: set ts=4 sw=4 sts=4 expandtab:
