#include "FairAlignmentHandler.h"

#include "FairLogger.h"

#include <TGeoManager.h>
#include <TGeoPhysicalNode.h>

FairAlignmentHandler::FairAlignmentHandler() {}

FairAlignmentHandler::~FairAlignmentHandler() {}

void FairAlignmentHandler::AlignGeometry() const
{
    if (fAlignmentMatrices.size() > 0) {
        LOG(info) << "aligning the geometry...";

        LOG(info) << "aligning in total " << fAlignmentMatrices.size() << " volumes.";
        if (gGeoManager->GetNAlignable() > 0) {
            AlignGeometryBySymlink();
        } else {
            AlignGeometryByFullPath();
        }

        LOG(info) << "alignment finished!";
    }
}

void FairAlignmentHandler::AlignGeometryByFullPath() const
{
    TString volume_path;

    LOG(info) << "aligning using full path.";
    for (auto const& alignment_entry : fAlignmentMatrices) {
        volume_path = alignment_entry.first;

        gGeoManager->cd(volume_path);

        TGeoNode* volume_node = gGeoManager->GetCurrentNode();
        TGeoMatrix* volume_matrix = volume_node->GetMatrix();
        // Need to do this as since ROOT 6.14 TGeoMatrix has no multiplication operator anymore
        // it is implimnted now in TGeoHMatrix

        TGeoHMatrix local_volume_matrix = TGeoHMatrix(*volume_matrix);

        TGeoHMatrix* new_volume_matrix = new TGeoHMatrix(local_volume_matrix * alignment_entry.second);
        // new matrix, representing real position (from new local mis RS to the global one)

        TGeoPhysicalNode* pn = gGeoManager->MakePhysicalNode(volume_path);

        pn->Align(new_volume_matrix);
    }
    LOG(info) << "alignments applied!";
}

void FairAlignmentHandler::AlignGeometryBySymlink() const
{
    TString volume_path;

    LOG(info) << "aligning using symlinks";
    for (auto const& alignment_entry : fAlignmentMatrices) {
        volume_path = alignment_entry.first;

        TGeoPhysicalNode* node = NULL;
        TGeoPNEntry* entry = gGeoManager->GetAlignableEntry(volume_path);
        if (entry) {
            node = gGeoManager->MakeAlignablePN(entry);
        }

        TGeoMatrix* volume_matrix = NULL;
        if (node) {
            volume_matrix = node->GetMatrix();
        } else {
            continue;
        }
        // Need to do this as since ROOT 6.14 TGeoMatrix has no multiplication operator anymore
        // it is implimnted now in TGeoHMatrix
        TGeoHMatrix local_volume_matrix = TGeoHMatrix(*volume_matrix);

        TGeoHMatrix* new_volume_matrix = new TGeoHMatrix(local_volume_matrix * alignment_entry.second);
        // new matrix, representing real position (from new local mis RS to the global one)
        node->Align(new_volume_matrix);
    }
}

void FairAlignmentHandler::AddAlignmentMatrices(const std::map<std::string, TGeoHMatrix>& alignmentMatrices,
                                                bool invertMatrices)
{
    LOG(info) << "adding inverting matrices...";
    for (auto const& m : alignmentMatrices) {
        if (invertMatrices)
            fAlignmentMatrices[m.first] *= m.second.Inverse();
        else
            fAlignmentMatrices[m.first] *= m.second;
    }
}
