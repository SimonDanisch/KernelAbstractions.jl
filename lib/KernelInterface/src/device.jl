"""
    get_global_size()::@NamedTuple{x::Int, y::Int, z::Int}

Return the number of global work-items specified.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_global_size()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_global_size end

"""
    get_global_id()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the unique global work-item ID.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_global_id()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_global_id end

"""
    get_local_size()::@NamedTuple{x::Int, y::Int, z::Int}

Return the number of local work-items specified.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_local_size()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_local_size end

"""
    get_local_id()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the unique local work-item ID.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_local_id()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_local_id end

"""
    get_num_groups()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the number of groups.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_num_groups()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_num_groups end

"""
    get_group_id()::@NamedTuple{x::Int, y::Int, z::Int}

Returns the unique group ID.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_group_id()::@NamedTuple{x::Int, y::Int, z::Int}
    ```
"""
function get_group_id end

"""
    get_sub_group_size()::UInt32

Returns the number of work-items in the sub-group.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_sub_group_size()::UInt32
    ```
"""
function get_sub_group_size end

"""
    get_max_sub_group_size()::UInt32

Returns the maximum sub-group size for sub-groups in the current workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_max_sub_group_size()::UInt32
    ```
"""
function get_max_sub_group_size end

"""
    get_num_sub_groups()::UInt32

Returns the number of sub-groups in the current workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_num_sub_groups()::UInt32
    ```
"""
function get_num_sub_groups end

"""
    get_sub_group_id()::UInt32

Returns the sub-group ID within the work-group.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_sub_group_id()::UInt32
    ```
"""
function get_sub_group_id end

"""
    get_sub_group_local_id()::UInt32

Returns the work-item ID within the current sub-group.

!!! note
    1-based.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override get_sub_group_local_id()::UInt32
    ```
"""
function get_sub_group_local_id end


"""
    localmemory(::Type{T}, dims)

Declare memory that is local to a workgroup.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override localmemory(::Type{T}, ::Val{Dims}) where {T, Dims}
    ```
    As well as the on-device functionality.
"""
localmemory(::Type{T}, dims) where {T} = localmemory(T, Val(dims))

# The `Val` form only exists in a backend's overlay method table, so off-device it
# would otherwise fall back to the forwarding method above and recurse forever.
localmemory(::Type{T}, ::Val) where {T} =
    error("Local memory used outside kernel or not captured")


"""
    shfl_down(val::T, offset::Integer) where T

Read `val` from a lane with higher id given by `offset`.

!!! note
    `shfl_down` must be encountered by all workitems of a sub-group executing the kernel or by none at all.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override shfl_down(val::T, offset::Integer) where T
    ```
    As well as the on-device functionality.

    This implementation **must** be synchronizing.
    That is, kernels using this function can safely assume that
    they do **not** need a `sub_group_barrier` before calling
    this function.
"""
function shfl_down end

"""
    shfl_down_types(::Backend)::Vector{DataType}

Returns a vector of `DataType`s supported on `backend`

!!! note
    Backend implementations **must** implement this function
    only if they support `shfl_down` for any types.
"""
shfl_down_types(::Backend) = DataType[]


"""
    sub_group_reduce_add(val::T) where T

Sum `val` across the sub-group and return the total on every lane.

Here rather than left to each backend because a portable kernel cannot build it
from [`shfl_down`](@ref) without giving up the hardware instruction: every
backend has one — SPIR-V's `OpGroupNonUniformFAdd`/`IAdd` with a `Reduce`
group operation, Metal's `simd_sum`, CUDA's `__reduce_add_sync` — and a
log2(subgroup) shuffle loop in its place is slower for the one reduction that
sits in the inner loop of every GEMV and every mapreduce.

!!! note
    `sub_group_reduce_add` must be encountered by all workitems of a sub-group
    executing the kernel or by none at all.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override sub_group_reduce_add(val::T) where T
    ```
    As well as the on-device functionality.

    This implementation **must** be synchronizing, on the same terms as
    [`shfl_down`](@ref): a kernel may assume it needs no `sub_group_barrier`
    before calling this function.
"""
function sub_group_reduce_add end

"""
    sub_group_reduce_add_types(::Backend)::Vector{DataType}

Returns a vector of `DataType`s supported on `backend`

!!! note
    Backend implementations **must** implement this function
    only if they support `sub_group_reduce_add` for any types.
"""
sub_group_reduce_add_types(::Backend) = DataType[]


"""
    barrier()

After a `barrier()` call, all read and writes to global and local memory
from each thread in the workgroup are visible in from all other threads in the
workgroup.

This does **not** guarantee that a write from a thread in a certain workgroup will
be visible to a thread in a different workgroup.

!!! note
    `barrier()` must be encountered by all workitems of a work-group executing the kernel or by none at all.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override barrier()
    ```
"""
function barrier()
    error("Group barrier used outside kernel or not captured")
end

"""
    sub_group_barrier()

After a `sub_group_barrier()` call, all read and writes to global and local memory
from each thread in the sub-group are visible in from all other threads in the
sub-group.

This does **not** guarantee that a write from a thread in a certain sub-group will
be visible to a thread in a different sub-group.

!!! note
    `sub_group_barrier()` must be encountered by all workitems of a sub-group executing the kernel or by none at all.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override sub_group_barrier()
    ```
"""
function sub_group_barrier()
    error("Sub-group barrier used outside kernel or not captured")
end

"""
    _print(args...)

    Overloaded by backends to enable `KernelAbstractions.@print`
    functionality.

!!! note
    Backend implementations **must** implement:
    ```
    @device_override _print(args...)
    ```
    If the backend does not support printing,
    define it to return `nothing`.

The generic fallback prints on the host, which keeps CPU backends working.
`Val` arguments are unwrapped, since `KernelAbstractions.@print` uses them to
pass literal strings through to backends that require compile-time format strings.
"""
@generated function _print(items...)
    args = []

    for i in 1:length(items)
        item = :(items[$i])
        T = items[i]
        if T <: Val
            item = QuoteNode(T.parameters[1])
        end
        push!(args, item)
    end

    return quote
        print($(args...))
    end
end
