#!/usr/bin/env python3

# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

"""!
@brief  Generates and updates the body of a C++ header with PDG codes and particle masses.
@author Vít Kučera <vit.kucera@cern.ch>, Inha University
@date   2023-09-21
"""

import os
from ctypes import c_bool
from enum import Enum

try:
    import ROOT  # pylint: disable=import-error
    from ROOT import o2
except (ModuleNotFoundError, ImportError) as exc:
    raise OSError("O2 environment is not loaded.") from exc


# Enum of PDG_t particles
class PdgROOT(Enum):
    kDown = ROOT.kDown
    kDownBar = ROOT.kDownBar
    kUp = ROOT.kUp
    kUpBar = ROOT.kUpBar
    kStrange = ROOT.kStrange
    kStrangeBar = ROOT.kStrangeBar
    kCharm = ROOT.kCharm
    kCharmBar = ROOT.kCharmBar
    kBottom = ROOT.kBottom
    kBottomBar = ROOT.kBottomBar
    kTop = ROOT.kTop
    kTopBar = ROOT.kTopBar
    kGluon = ROOT.kGluon
    kElectron = ROOT.kElectron
    kPositron = ROOT.kPositron
    kNuE = ROOT.kNuE
    kNuEBar = ROOT.kNuEBar
    kMuonMinus = ROOT.kMuonMinus
    kMuonPlus = ROOT.kMuonPlus
    kNuMu = ROOT.kNuMu
    kNuMuBar = ROOT.kNuMuBar
    kTauMinus = ROOT.kTauMinus
    kTauPlus = ROOT.kTauPlus
    kNuTau = ROOT.kNuTau
    kNuTauBar = ROOT.kNuTauBar
    kGamma = ROOT.kGamma
    kZ0 = ROOT.kZ0
    kWPlus = ROOT.kWPlus
    kWMinus = ROOT.kWMinus
    kPi0 = ROOT.kPi0
    kK0Long = ROOT.kK0Long
    kPiPlus = ROOT.kPiPlus
    kPiMinus = ROOT.kPiMinus
    kProton = ROOT.kProton
    kProtonBar = ROOT.kProtonBar
    kNeutron = ROOT.kNeutron
    kNeutronBar = ROOT.kNeutronBar
    kK0Short = ROOT.kK0Short
    kK0 = ROOT.kK0
    kK0Bar = ROOT.kK0Bar
    kKPlus = ROOT.kKPlus
    kKMinus = ROOT.kKMinus
    kLambda0 = ROOT.kLambda0
    kLambda0Bar = ROOT.kLambda0Bar
    kLambda1520 = ROOT.kLambda1520
    kSigmaMinus = ROOT.kSigmaMinus
    kSigmaBarPlus = ROOT.kSigmaBarPlus
    kSigmaPlus = ROOT.kSigmaPlus
    kSigmaBarMinus = ROOT.kSigmaBarMinus
    kSigma0 = ROOT.kSigma0
    kSigma0Bar = ROOT.kSigma0Bar
    kXiMinus = ROOT.kXiMinus
    kXiPlusBar = ROOT.kXiPlusBar
    kOmegaMinus = ROOT.kOmegaMinus
    kOmegaPlusBar = ROOT.kOmegaPlusBar


# Enum of additional particles
class Pdg(Enum):
    kEta = 221
    kOmega = 223
    kEtaPrime = 331
    kB0 = 511
    kB0Bar = -511
    kBPlus = 521
    kBCPlus = 541
    kBS = 531
    kBSBar = -531
    kD0 = 421
    kD0Bar = -421
    kD0StarPlus = 10411
    kD0Star0 = 10421
    kD1Plus = 20413
    kD10 = 20423
    kD2StarPlus = 415
    kD2Star0 = 425
    kDMinus = -411
    kDPlus = 411
    kDS = 431
    kDSBar = -431
    kDSStar = 433
    kDS1 = 10433
    kDS1Star2700 = 30433
    kDS1Star2860 = 40433
    kDS2Star = 435
    kDS3Star2860 = 437
    kDStar = 413
    kDStar0 = 423
    kChiC1 = 20443
    kJPsi = 443
    kLambdaB0 = 5122
    kLambdaCPlus = 4122
    kLambdaCPlus2860 = 24124
    kLambdaCPlus2880 = 24126
    kLambdaCPlus2940 = 4125
    kOmegaC0 = 4332
    kK0Star892 = 313
    kKPlusStar892 = 323
    kPhi = 333
    kSigmaC0 = 4112
    kSigmaCPlusPlus = 4222
    kSigmaCStar0 = 4114
    kSigmaCStarPlusPlus = 4224
    kX3872 = 9920443
    kXi0 = 3322
    kXiB0 = 5232
    kXiCCPlusPlus = 4422
    kXiCPlus = 4232
    kXiC0 = 4132
    kXiC3055Plus = 4325
    kXiC3080Plus = 4326
    kXiC3055_0 = 4315
    kXiC3080_0 = 4316
    kDeuteron = 1000010020
    kTriton = 1000010030
    kHelium3 = 1000020030
    kAlpha = 1000020040
    kLithium4 = 1000030040
    kHyperTriton = 1010010030
    kHyperHydrogen4 = 1010010040
    kHyperHelium4 = 1010020040
    kHyperHelium5 = 1010020050
    kHyperHelium4Sigma = 1110020040
    kLambda1520_Py = 102134  # PYTHIA code different from PDG
    kK1_1270_0 = 10313
    kK1_1270Plus = 10323


dbPdg = o2.O2DatabasePDG


def mass(code):
    """Returns particle mass from o2::O2DatabasePDG."""
    # Missing particles should be added in O2DatabasePDG.h.
    success = c_bool(True)
    return dbPdg.Mass(code, success)


def declare_mass(pdg, mass_type="double") -> str:
    """Returns a C++ declaration of a particle mass constant."""
    return f"constexpr {mass_type} Mass{pdg.name[1:]} = {mass(pdg.value)};"


def main():
    """Main function"""

    path_header = "PhysicsConstants.h"
    name_script = os.path.basename(__file__)

    # Comment at the beginning of the output
    block_begin = "// BEGINNING OF THE GENERATED BLOCK."
    # Comment at the end of the output
    block_end = "// END OF THE GENERATED BLOCK"
    # Preamble with instructions
    block_preamble = (
        "// DO NOT EDIT THIS BLOCK DIRECTLY!"
        f"\n// It has been generated by the {name_script} script."
        "\n// For modifications, edit the script and generate this block again."
    )
    # Start of enum declarations of additional particles
    enum_o2_head = (
        "/// \\brief Declarations of named PDG codes of particles missing in ROOT PDG_t"
        "\n/// \\note Follow kCamelCase naming convention"
        "\n/// \\link https://root.cern/doc/master/TPDGCode_8h.html"
        "\nenum Pdg {"
    )
    # End of enum declarations of additional particles
    enum_o2_foot = "};"
    # Documentation string for mass declarations of additional particles
    mass_o2_head = "/// \\brief Declarations of masses for additional particles"
    # Documentation string for mass declarations of PDG_t particles
    mass_root_head = "/// \\brief Declarations of masses for particles in ROOT PDG_t"

    # Get header content before and after the generated block.
    print(f'File "{path_header}" will be updated.')
    try:
        with open(path_header, encoding="utf-8") as file:
            content_old = file.readlines()
    except OSError as exc:
        raise OSError(f'Failed to open file "{path_header}".') from exc
    lines_header_before: list[str] = []
    lines_header_after: list[str] = []
    got_block_begin = False
    got_block_end = False
    for line in content_old:
        line = line.strip()
        if line == block_begin:
            got_block_begin = True
        if not got_block_begin:
            lines_header_before.append(line)
        if got_block_end:
            lines_header_after.append(line)
        if line == block_end:
            got_block_end = True
    if not got_block_begin:
        raise ValueError("Did not find the beginning of the block.")
    if not got_block_end:
        raise ValueError("Did not find the end of the block.")

    # Additional particles
    lines_enum_o2: list[str] = [enum_o2_head]
    lines_mass_o2: list[str] = [mass_o2_head]
    for pdg_o2 in Pdg:
        lines_enum_o2.append(f"  {pdg_o2.name} = {pdg_o2.value},")
        lines_mass_o2.append(declare_mass(pdg_o2))
    lines_enum_o2[-1] = lines_enum_o2[-1][:-1]  # Remove the last comma.
    lines_enum_o2.append(enum_o2_foot)

    # PDG_t particles
    lines_mass_root: list[str] = [mass_root_head]
    for pdg_root in PdgROOT:
        lines_mass_root.append(declare_mass(pdg_root))

    # Header body
    content_new = "\n".join(
        (
            *lines_header_before,
            block_begin,
            block_preamble,
            "",
            *lines_enum_o2,
            "",
            *lines_mass_o2,
            "",
            *lines_mass_root,
            "",
            block_end,
            *lines_header_after,
            "",
        )
    )
    # print(content_new)

    # Overwrite the input file.
    try:
        with open(path_header, "w", encoding="utf-8") as file:
            file.write(content_new)
            print(f'File "{path_header}" has been overwritten.')
    except OSError as exc:
        raise OSError(f'Failed to write to file "{path_header}".') from exc


if __name__ == "__main__":
    main()
