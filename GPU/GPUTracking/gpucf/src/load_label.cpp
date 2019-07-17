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

    SectorData<RawLabel> rawLabels = gpucf::read<RawLabel>(args::get(infile));

    for (size_t i = 0; i < 10; i++)
    {
        log::Debug() << rawLabels.data[i].id;
    }

    gpucf::log::Info() << "Creating label container";

    SectorMap<LabelContainer> container = LabelContainer::bySector(rawLabels);

    std::unordered_map<MCLabel, size_t> digitsPerTrack;

    for (const LabelContainer &labels : container)
    {
        for (size_t i = 0; i < labels.size(); i++) 
        {
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
    }

    for (auto it : digitsPerTrack)
    {
        log::Info() << it.first << ": " << it.second;
    }


    return 0;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
