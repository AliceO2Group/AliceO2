# ALICE-O2 GEANT Simulation of CAD Geometries

These are a few notes related to the inclusion of external (CAD-described) detector modules into the O2 simulation framework.

## Description of the Workflow

In principle, such integration is now possible and requires the following steps:

1. The CAD geometry needs to be exported to STEP format and must contain only the final geometry (no artificial eta-cut elements). Ideally, the geometry should be fully hierarchical with proper solid reuse. The solids should retain their proper surface representation for detailed analysis. Materials can be treated by providing a CSV file that map STEP part names to a material name. The conversion code will do it's best to find a corresponding material definition from a G4 NIST database JSON file (which can be expanded by users with custom definitions).


2. A tool `O2-CADtoTGeo.py` is provided to convert the STEP geometry into TGeo format. The tool is part of AliceO2 and is based on Python bindings (OCC) for OpenCascade. The tool can be used as follows:

    ```bash
    python O2-CADtoTGeo.py STEP_FILE --output-folder my_detector -o geom.C --mesh \
                           --mesh-prec 0.2
    ```

    This will create a ROOT macro file `geom.C` containing the geometry description in ROOT format, as well as several binary files describing the TGeo solids. The `geom.C` file can either be used directly in ROOT to inspect the geometry or be provided to ALICE-O2 for inclusion in the geometry.

    When materials are included the conversion process looks like this
    ```bash
    python O2-CADtoTGeo.py STEP_FILE --output-folder my_detector -o geom.C --mesh \
                           --mesh-prec 0.2                                        \
                           --materials-csv MATERIALS.csv                          \ --g4-nist-json ../g4_nist_database/G4_NIST_DB.json 
    ```

3. Inspection of the created geom.C file and possible manual editing/fixing of the code, in particular materials and medium objects.

4. Once the conversion is complete, the module can be inserted into the O2 geometry via the `ExternalModule` class. To do so, follow this pattern in `build_geometry.C`:

    ```cpp
    if (isActivated("EXT")) {
      o2::passive::ExternalModuleOptions options;
      options.root_macro_file = "PATH_TO_MY_DETECTOR/my_detector/geom_withMaterials.C";
      options.anchor_volume = "barrel"; // hook this into barrel
      auto rot = new TGeoCombiTrans();
      rot->RotateX(90);
      rot->SetDy(30); // compensate for a shift of the barrel with respect to zero
      options.placement = rot;
      run->AddModule(new o2::passive::ExternalModule("A3VTX", "ALICE3 beam pipe", options));
    }
    ```

5. Create a custom detector geometry list file `my_det.json` in JSON format that includes the external detector (and any other required components, such as the L3 magnet in this example):

    ```json
    {
      "MY_DET": [
        "EXT",
        "MAG"
      ]
    }
    ```

6. Run the Geant simulation with:

   ```bash
   o2-sim --detectorList MY_DET:my_det.json -g pythia8pp ....
   ```

## Known Limitations

- The `O2-CADtoTGeo.py` tool currently converts geometries only into TGeoTessellated solids. This may be suboptimal for primitive shapes or only an approximation for shapes with exact second-order surfaces (e.g., tubes). The precision (and therefore the number of surface triangles) can be controlled with the `--mesh-prec` parameter. The smaller the value, the more precise the mesh.

- Meshed solids created by the tool may have issues, such as topological errors or non-watertight surfaces. It is planned to include "healing" steps via additional processing with well-known geometry kernels (e.g., CGAL).

- The tool does not currently export materials or TGeoMedia. These must be inserted or edited manually. It is planned to make this process more automatic and user-friendly.

- The Python tool requires the OCC Python module, which is currently not part of our software distribution. We have found it most practical to run the tool in a separate conda environment (fully decoupled from the ALICE software stack).

- The tool currently generates a `geom.C` macro file. In the future, it may be possible to directly create an in-memory TGeo representation for deeper integration.

- Currently, only passive modules can be integrated. Treatment of sensitive volumes or parts will be addressed in a future step.

## Software Installation

- The simulation must be run in the standard O2 environment built with alibuild.

- The CAD conversion tool must currently be run in a dedicated conda environment, as described in scripts/geometry/README.md in the AliceO2 source code.