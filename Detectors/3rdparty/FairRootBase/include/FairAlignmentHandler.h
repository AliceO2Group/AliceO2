#ifndef FAIR_ALIGNMENTHANDLER_H
#define FAIR_ALIGNMENTHANDLER_H

#include <TGeoMatrix.h>
#include <map>

class FairAlignmentHandler
{
    friend class FairRun;
    std::map<std::string, TGeoHMatrix> fAlignmentMatrices;

    void AlignGeometry() const;
    void AlignGeometryByFullPath() const;
    void AlignGeometryBySymlink() const;

    void AddAlignmentMatrices(const std::map<std::string, TGeoHMatrix>& alignmentMatrices, bool invertMatrices);

  public:
    FairAlignmentHandler();
    virtual ~FairAlignmentHandler();
};

#endif
