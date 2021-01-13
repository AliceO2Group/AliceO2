
o2_add_library(
  DCStestWorkflow
  SOURCES src/DCSRandomDataGeneratorSpec.cxx
  PUBLIC_LINK_LIBRARIES O2::DetectorsDCS O2::Framework)

o2_add_executable(
  data-workflow
  COMPONENT_NAME dcs
  SOURCES src/dcs-data-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow)

o2_add_executable(
  random-data-workflow
  COMPONENT_NAME dcs
  SOURCES src/dcs-random-data-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow)

o2_add_executable(
  proxy
  COMPONENT_NAME dcs
  SOURCES src/dcs-proxy.cxx
  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow)

o2_add_executable(
  data-client
  COMPONENT_NAME dcs
  SOURCES src/dcs-data-client-workflow.cxx
  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow)

