# Exercise 10: Weather Router Test Cases

## Assignment

Write four valid-input test cases for a router with illuminance (millilux),
highway water level (micrometers), and temperature (degrees Celsius). The
expected result is yes/no; negative testing is not required. Use a five-column
table: ID, three inputs, expected output.

## Existing Evidence

The router returns yes when the neural-network path should be activated:
illuminance below 500 millilux OR (water level above 1000 micrometers AND
temperature below 0 degrees Celsius).

| Test case ID | Illuminance (millilux) | Water level (um) | Temperature (C) | Expected output |
| --- | ---: | ---: | ---: | --- |
| R-01 | 499 | 1000 | 0 | True |
| R-02 | 500 | 1001 | -1 | True |
| R-03 | 500 | 1000 | -1 | False |
| R-04 | 500 | 1001 | 0 | False |

Here yes means the NN route is activated, not that a speed reduction or safe
outcome is guaranteed. These four cases exercise the strict boundaries and
distinguish the OR router rule from an incorrect AND rule. The automated test
suite also checks all combinations of darkness, wetness, and freezing and
additional boundary and invalid-input cases.
