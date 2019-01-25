#include "DataSet.h"

#include <filesystem/path.h>

#include <fstream>


using namespace gpucf;
namespace fs = filesystem;


void DataSet::read(const fs::path &file)
{
    objs.clear();

    std::ifstream in(file.str());

    for (std::string line;
         std::getline(in, line);)
    {
        nonstd::optional<Object> obj = Object::tryParse(line);

        if (obj.has_value())
        {
            objs.push_back(*obj);    
        }
    }
}

void DataSet::write(const fs::path &file) const
{
    std::ofstream out(file.str());

    for (const Object &obj : objs)
    {
        out << obj.str() << "\n";
    }
}

std::vector<Object> DataSet::get() const
{
    return objs;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
