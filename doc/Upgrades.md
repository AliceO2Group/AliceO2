# Detector upgrades in O2

## Conventions

Code for detector upgrades shall be developed in `Detectors/Upgrades` and
`DataFormats/Detectors/Upgrades`. It is compiled only if the CMake option
`ENABLE_UPGRADES` is activated, e.g.
```sh
cmake -DENABLE_UPGRADES=ON <src dir>
```
Upgrade-only CMake instructions should be protected as:
```cmake
if(ENABLE_UPGRADES)
# ...
endif(ENABLE_UPGRADES)
```
Individual pieces of code to be compiled only when upgrades are built are
to be guarded by precompiler macros:
```c++
#ifdef ENABLE_UPGRADES
// ...
#endif
```
The corresponding compiler definition is added automatically and globally.
