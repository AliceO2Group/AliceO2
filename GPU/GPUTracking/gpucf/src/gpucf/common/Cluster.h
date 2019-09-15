// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Object.h>

#include <shared/ClusterNative.h>

#include <iosfwd>


namespace gpucf
{

class Cluster 
{

public:
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

    float Q;
    float QMax;
    float padMean;
    float timeMean;
    float padSigma;
    float timeSigma;
    int cru;
    int row;

    bool atEdge      = false;
    bool splitInTime = false;
    bool splitInPad  = false;

    Cluster();
    Cluster(int, int, float, float, float, float, float, float);
    Cluster(const Digit &, const ClusterNative &);
    Cluster(int, int, const ClusterNative &);
    Cluster(int, const ClusterNative &);

    Object serialize() const;
    void deserialize(const Object &);
    bool hasNaN() const;
    bool hasNegativeEntries() const;

    int globalRow() const;

    bool operator==(const Cluster &) const;

    bool eq(const Cluster &, float, float, FieldMask) const;

    float dist(const Cluster &) const;
};

std::ostream &operator<<(std::ostream &, const Cluster &);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
