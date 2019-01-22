#pragma once


namespace gpucf
{

template<typename T>
class Dataset
{

public:


    Dataset(const std::vector<T> &d)
        : data(d)
    {
    }

    std::string serialize() const;

private:
    std::vector<T> data;



};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
