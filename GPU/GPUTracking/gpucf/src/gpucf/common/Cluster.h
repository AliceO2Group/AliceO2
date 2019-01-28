#pragma once

#include <gpucf/common/Object.h>

#include <shared/Cluster.h>

#include <iosfwd>
#include <string>


namespace gpucf
{

class Cluster : public FloatCluster
{

public:
    static constexpr size_t floatMemberNum = 6;

    using FieldMask = unsigned char;

    enum Field : FieldMask
    { 
        Field_Q         = (1 << 0),
        Field_QMax      = (1 << 1),
        Field_timeMean  = (1 << 2),
        Field_padMean   = (1 << 3),
        Field_timeSigma = (1 << 4),
        Field_padSigma  = (1 << 5),

        Field_all = Field_Q 
                  | Field_QMax 
                  | Field_timeMean 
                  | Field_padMean 
                  | Field_timeSigma 
                  | Field_padSigma
    };

    Cluster();
    Cluster(int, int, float, float, float, float, float, float);

    Object serialize() const;
    void deserialize(const Object &);

    bool hasNaN() const;
    bool hasNegativeEntries() const;

    int globalRow() const;

    bool operator==(const Cluster &) const;

    bool eq(const Cluster &, float, float, FieldMask) const;
};

std::ostream &operator<<(std::ostream &, const Cluster &);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
