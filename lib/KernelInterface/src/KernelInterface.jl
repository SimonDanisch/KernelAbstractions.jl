"""
# `KernelInterface`

The `KernelInterface` (or `KI`) module defines the API interface for backends to define various lower-level device and
host-side functionality. The `KI` interface is used to define the higher-level device-side
functionality in `KernelAbstractions`.

Both provide APIs for host and device-side functionality, but `KI` focuses on lower-level
functionality that is shared amongst backends, while `KernelAbstractions` provides higher-level functionality
such as writing kernels that work on arrays with an arbitrary number of dimensions, or convenience functions
like allocating arrays on a backend.
"""
module KernelInterface

include("utils.jl")

include("backend.jl")
include("device.jl")
include("launch.jl")
include("host.jl")
# `caps.jl` after `matrix.jl`: `DeviceCaps` holds a `Vector{MatrixShape}`.
include("matrix.jl")
include("caps.jl")
# The cooperative-matrix type and the nine operations a backend lowers.
include("coopmat.jl")
# Graphics vocabulary a compiler and a backend both dispatch on, for the same
# reason the matrix vocabulary is here.
include("topology.jl")
# The device-side names a shader body calls. Here rather than in a runtime,
# because a compiler lowers them and a runtime only launches what was compiled
# — and because a backend cannot override a name declared in a package that
# depends on IT. See the headers of both files.
include("graphics.jl")
include("raytracing.jl")

end
