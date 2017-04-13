# BLAS implementation for D

This BLAS implementation for D is based on the [GNU Scientific Library](https://www.gnu.org/software/gsl/manual/) BLAS module. This is the first version release of the library and much of the code was created by code conversion of GSL. At present only the template functions are available
and not the type specific aliases - however all the relevant functionality can be obtained from the template functions.

[MKL BLAS](https://software.intel.com/en-us/articles/mkl-reference-manual) documentation is being used as a placeholder with a view to overwrite later.

## Next Phase

1. Performance optimization
2. Complete unit test coverage
3. Type specific BLAS aliases

## Example

Executing a single file

```
#!/usr/bin/env dub
/+ dub.json:
{
    "name": "testdblas",
    "dependencies": {"dblas": "~>0.0.1"},
}
+/

/*
*  Compile example:
*  dub run --single testdblas.d
*/

import std.stdio : writeln;
import dblas;
import std.math: complex;

void main()
{
	double[] x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
  auto y = [complex(1.0, 1.5), complex(2.0, 2.5), complex(3.0, 3.5), complex(4.0, 4.5), 
           complex(5.0, 5.5), complex(6.0, 6.5), complex(7.0, 7.5)];
  scal(x.length, 3.0, x.ptr, 1);
  scal(y.length, 3.0, y.ptr, 1);
  writeln("scal output:", x);
  writeln("scal output:", y);
  writeln("asum output: ", asum(x.length, x.ptr, 1));
  writeln("asum output: ", asum(y.length, y.ptr, 1));
}
```

