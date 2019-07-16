#include <gpucf/common/LabelContainer.h>
#include <gpucf/common/log.h>
#include <gpucf/common/serialization.h>

#include <args/args.hxx>

#include <iostream>


using namespace gpucf;


int main(int argc, const char *argv[])
{
    args::ArgumentParser parser("");

    args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
    args::ValueFlag<std::string> infile(parser, "IN", "Label file", {'i', "in"});

    try 
    {
        parser.ParseCLI(argc, argv);
    } 
    catch (const args::Help &) 
    {
        std::cerr << parser;
        return 1;
    }

    gpucf::log::Info() << "Reading label file " << args::get(infile);

    std::vector<RawLabel> rawlabels = gpucf::read<RawLabel>(args::get(infile));

    gpucf::log::Info() << "Creating label container";

    LabelContainer labels(rawlabels);
    std::unordered_map<MCLabel, size_t> digitsPerTrack;

    for (size_t i = 0; i < labels.size(); i++) {
        /* log::Info() << i << ": " << labels[i].size(); */

        for (const MCLabel &l : labels[i])
        {
            auto lookup = digitsPerTrack.find(l);
            if (lookup == digitsPerTrack.end())
            {
                digitsPerTrack[l] = 1;
            }
            else
            {
                lookup->second++;
            }
        }
    }

    for (auto it : digitsPerTrack)
    {
        log::Info() << it.first << ": " << it.second;
    }


    return 0;
}



// vim: set ts=4 sw=4 sts=4 expandtab:

