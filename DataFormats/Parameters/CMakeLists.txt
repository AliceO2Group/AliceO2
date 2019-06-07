o2_add_library(DataFormatsParameters
               SOURCES
               src/GRPObject.cxx
               PUBLIC_LINK_LIBRARIES
               FairRoot::Base
               O2::CommonConstants
               O2::CommonTypes
               O2::DetectorsCommonDataFormats)

o2_target_root_dictionary(DataFormatsParameters HEADERS
                          include/DataFormatsParameters/GRPObject.h LINKDEF
                          src/ParametersDataLinkDef.h)
# note we are explicitely giving the LINKDEF parameter as the LinkDef does not
# follow the usual naming scheme [module]LinkDef.h
#
# * should be src/DataFormatParametersLinkDef.h
# * is src/ParametersDataLinkDef.h instead
