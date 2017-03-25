# BLAS implementation for D

This BLAS implementation for D is currently based on the GNU Scientific Library BLAS module. The library is currently in the first phase - code conversion for GSL. Documentation of the function is largely taken from the [MKL manual](https://software.intel.com/en-us/articles/mkl-reference-manual). Once the code conversion is completed efforts will be made to  optimise performance.

The approach taken in writing the code is to use D's metaprogramming features as much as possible to reduce code complexity.

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

