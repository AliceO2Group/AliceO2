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

o2_add_library(ForwardAlign
        SOURCES src/MatrixSparse.cxx
                src/MatrixSq.cxx
                src/MillePede2.cxx
                src/MillePedeRecord.cxx
                src/MilleRecordReader.cxx
                src/MilleRecordWriter.cxx
                src/MinResSolve.cxx
                src/RectMatrix.cxx
                src/SymBDMatrix.cxx
                src/SymMatrix.cxx
                src/VectorSparse.cxx
                src/MilleRecordWriterSpec.cxx
        PUBLIC_LINK_LIBRARIES O2::CCDB
                O2::Steer
                ROOT::TreePlayer)

o2_target_root_dictionary(ForwardAlign
        HEADERS include/ForwardAlign/MatrixSparse.h
                include/ForwardAlign/MatrixSq.h
                include/ForwardAlign/MillePede2.h
                include/ForwardAlign/MillePedeRecord.h
                include/ForwardAlign/MilleRecordReader.h
                include/ForwardAlign/MilleRecordWriter.h
                include/ForwardAlign/MinResSolve.h
                include/ForwardAlign/RectMatrix.h
                include/ForwardAlign/SymBDMatrix.h
                include/ForwardAlign/SymMatrix.h
                include/ForwardAlign/VectorSparse.h
                include/ForwardAlign/MilleRecordWriterSpec.h
        LINKDEF src/ForwardAlignLinkDef.h)



o2_add_executable(
        millerecord-writer-workflow
        SOURCES src/MilleRecordWriterSpec.cxx src/millerecord-writer-workflow.cxx
        COMPONENT_NAME fwdalign
        PUBLIC_LINK_LIBRARIES O2::Framework O2::DPLUtils O2::ReconstructionDataFormats O2::SimulationDataFormat O2::ForwardAlign)
