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
# The cooperative-matrix type and the eleven operations a backend lowers.
include("coopmat.jl")
# Loading and storing a matrix through a description of the array in memory.
include("tensor.jl")
# Graphics vocabulary a compiler and a backend both dispatch on, for the same
# reason the matrix vocabulary is here.
include("topology.jl")
# The device-side names a shader body calls. Here rather than in a runtime,
# because a compiler lowers them and a runtime only launches what was compiled
# — and because a backend cannot override a name declared in a package that
# depends on IT. See the headers of both files.
include("graphics.jl")
include("raytracing.jl")
# The mesh pipeline, after `graphics.jl`: its emitter is what a geometry body
# emits through, and the native lowering of that reaches the geometry stage
# intrinsics declared there.
include("mesh.jl")

# the public API; nothing is exported, so that `KI.` prefixes the interface everywhere
@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public,
            # backends
            :Backend, :get_backend,
            # device side
            :get_global_id, :get_global_size, :get_local_id, :get_local_size,
            :get_group_id, :get_num_groups,
            :get_sub_group_size, :get_max_sub_group_size, :get_num_sub_groups,
            :get_sub_group_id, :get_sub_group_local_id,
            :localmemory, :shfl_down, :barrier, :sub_group_barrier, :_print,
            # compilation and launch
            :Kernel, :kernel_function, :argconvert, :launch, Symbol("@launch"),
            :launch_configuration, :max_work_group_size, :max_work_group_dims,
            :max_num_groups, :sub_group_size, :multiprocessor_count,
            # host side
            :allocate, :zeros, :ones, :copyto!, :pagelock!, :unsafe_free!,
            :synchronize, :record_event, :wait_event, :priority!,
            :device, :ndevices, :device!,
            :functional, :versioninfo,
            :supports_unified, :supports_atomics, :supports_float64,
            :supports_subgroups, :supports_shuffle,
            # cooperative-matrix extensions and tensor addressing
            :coopmat_perelement, :coopmat_reduce, :CoopMatReduce,
            :supports_coopmat_perelement, :supports_coopmat_reduce,
            :supports_flexible_coopmat_shapes,
            :TensorLayout, :TensorView, :TENSOR_CLAMP_UNDEFINED, :TENSOR_CLAMP_CONSTANT,
            :TENSOR_CLAMP_TO_EDGE, :tensor_layout, :tensor_setdim, :tensor_setstride,
            :tensor_setclampvalue, :tensor_clampbits, :tensor_slice, :tensor_view,
            :tensor_load, :tensor_store, :supports_tensor_addressing,
        )
    )
end

end
