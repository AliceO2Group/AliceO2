By default the tests are running without any input.

If you want to go further though, you can input a JSON file containing test positions using the  `--testpos` option (it is an option of the test module, not of the test program itself, hence the `--` syntax) and call (part of) the test like so :  

```
./test_MCHMappingTest --run_test="*/*/*TestPos*" -- --testpos /some/dir/to/test_random_pos.json
````