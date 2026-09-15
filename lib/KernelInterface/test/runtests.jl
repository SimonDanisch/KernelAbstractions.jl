# These are the standalone tests for KernelInterface

using KernelInterface
using Aqua
using Test

const KI = KernelInterface

# `_print`'s host fallback writes to `stdout`, so capture it through a real file.
function capture_stdout(f)
    return mktemp() do path, io
        redirect_stdout(f, io)
        flush(io)
        return read(path, String)
    end
end

@testset "standalone" begin
    # KernelInterface is what backends implement against, so it must stay loadable
    # without dragging in KernelAbstractions or a compiler stack.
    toml = read(joinpath(pkgdir(KernelInterface), "Project.toml"), String)
    @test !occursin("[deps]", toml)
    @test !occursin("[sources]", toml)
end

# NOTE: this runs before the mock backend below defines methods on `argconvert`
# and `kernel_function`.
@testset "interface stubs" begin
    # These have no fallback on purpose: a backend that forgets to `@device_override`
    # them should get a MethodError rather than silently wrong behaviour.
    stubs = [
        KI.get_global_size, KI.get_global_id,
        KI.get_local_size, KI.get_local_id,
        KI.get_num_groups, KI.get_group_id,
        KI.get_sub_group_size, KI.get_max_sub_group_size,
        KI.get_num_sub_groups, KI.get_sub_group_id,
        KI.get_sub_group_local_id,
        KI.shfl_down, KI.sub_group_reduce_add,
        KI.kernel_max_work_group_size, KI.max_work_group_size, KI.sub_group_size,
        KI.argconvert, KI.kernel_function,
        # Host-side stubs: required backend methods with no sensible fallback.
        KI.synchronize, KI.copyto!,
        # The one capability query everything in `caps.jl` derives from. A
        # default here would be a plausible fake device, which is worse than a
        # MethodError naming the backend that has not answered.
        KI.caps,
    ]
    for stub in stubs
        @test isempty(methods(stub))
    end
end

struct StubBackend <: KI.Backend end

# An array type with a known backend, for exercising the `get_backend` fallback
# that unwraps wrapper arrays.
struct BackedArray{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
end
Base.size(A::BackedArray) = size(A.data)
Base.getindex(A::BackedArray{T, N}, i::Vararg{Int, N}) where {T, N} = A.data[i...]
KI.get_backend(::BackedArray) = StubBackend()

# A backend implementing only `allocate`, as the interface requires.
struct AllocBackend <: KI.Backend end
function KI.allocate(::AllocBackend, ::Type{T}, dims::Tuple; unified::Bool = false) where {T}
    return Array{T}(undef, dims)
end

@testset "host fallbacks" begin
    # Barriers are meaningless off-device and must say so rather than no-op.
    @test_throws "used outside kernel" KI.barrier()
    @test_throws "used outside kernel" KI.sub_group_barrier()

    # Permissive defaults: a backend only implements these if it can do better.
    @test KI.shfl_down_types(StubBackend()) == DataType[]
    @test KI.sub_group_reduce_add_types(StubBackend()) == DataType[]
    @test KI.multiprocessor_count(StubBackend()) == 0

    # `localmemory` forwards the untyped `dims` to the `Val` form backends override.
    # Off-device that form is unimplemented, and must error rather than recurse
    # back into the forwarding method.
    @test_throws "used outside kernel" KI.localmemory(Float32, (2, 2))
    @test_throws "used outside kernel" KI.localmemory(Float32, Val((2, 2)))

    # The id is what makes two buffers of one type and shape two buffers: a
    # backend lowers workgroup memory to a global keyed by what it is given, so
    # without it the second allocation IS the first and the second write lands
    # on the first tile. Silent, and wrong. Every form has to reach the `Val`
    # method a backend overrides, which is what these pin.
    @test_throws "used outside kernel" KI.localmemory(Float32, (2, 2), 7)
    @test only(methods(KI.localmemory, Tuple{Type{Float32}, Val, Val})).nargs == 4
    # Two arguments is the one-buffer case and defaults the id, so it stays.
    @test hasmethod(KI.localmemory, Tuple{Type{Float32}, Tuple{Int, Int}})
end

@testset "get_backend" begin
    @test KI.GPU <: KI.Backend

    # The fallback finds the backend of wrapper arrays by walking `parent`.
    arr = BackedArray([1, 2, 3])
    @test KI.get_backend(arr) === StubBackend()
    @test KI.get_backend(view(arr, 1:2)) === StubBackend()
    @test KI.get_backend(reshape(arr, 3, 1)) === StubBackend()
    @test KI.get_backend(reinterpret(UInt, arr)) === StubBackend()

    # An array that is its own parent has no wrapped backend to find; the
    # fallback must error rather than recurse.
    @test_throws ArgumentError KI.get_backend([1, 2, 3])
end

@testset "backend queries" begin
    b = StubBackend()

    # `versioninfo` falls back to printing a notice, defaulting to `stdout`.
    @test occursin("not implemented", sprint(KI.versioninfo, b))
    @test occursin("not implemented", capture_stdout(() -> KI.versioninfo(b)))

    # `missing` distinguishes "not implemented" from a definite yes/no.
    @test KI.functional(b) === missing

    # Single-device defaults; `device!` still bounds-checks the id.
    @test KI.device(b) == 1
    @test KI.ndevices(b) == 1
    @test KI.device!(b, 1) === nothing
    @test_throws ArgumentError KI.device!(b, 0)
    @test_throws ArgumentError KI.device!(b, 2)

    # `priority!` validates the symbol even when the backend ignores it.
    for prio in (:high, :normal, :low)
        @test KI.priority!(b, prio) === nothing
    end
    @test_throws "priority must be one of" KI.priority!(b, :bogus)

    # Capability defaults: pessimistic for unified memory, optimistic otherwise.
    @test KI.supports_unified(b) === false
    @test KI.supports_atomics(b) === true
    @test KI.supports_float64(b) === true

    # Pinning is optional and freeing is a no-op unless a backend does better.
    @test KI.pagelock!(b, zeros(2)) === missing
    @test KI.unsafe_free!(zeros(2)) === nothing
end

# One real card's cooperative-matrix table: four entries share `16x16x16` and
# differ only in type, which is why `MatrixShape` carries `ab`/`acc` and why
# matching on extents alone is wrong.
const THISCARD = [
    KI.MatrixShape(Float16, Float32, 16, 16, 16, KI.SubgroupScope()),
    KI.MatrixShape(Float16, Float16, 16, 16, 16, KI.SubgroupScope()),
    KI.MatrixShape(UInt8, Int32, 16, 16, 16, KI.SubgroupScope()),
    KI.MatrixShape(Int8, Int32, 16, 16, 16, KI.SubgroupScope()),
]

# A backend with matrix hardware, for the queries that go through `caps`. Note it
# answers ONE question — everything else derives, which is the point of putting
# `DeviceCaps` here instead of in each backend.
struct MatrixBackend <: KI.Backend end
KI.caps(::MatrixBackend) = KI.DeviceCaps(
    true, 16, 32, 32, 49152, 1024, 48, 48,
    [(32, 16, 16, 16), (128, 32, 16, 16), (256, 32, 32, 16)], THISCARD
)

# The same device with cooperative matrices switched off, which is what a copy
# with `coopmat = false` means. It still CARRIES the table.
struct NoMatrixBackend <: KI.Backend end
KI.caps(::NoMatrixBackend) = KI.DeviceCaps(KI.caps(MatrixBackend()); coopmat = false)

@testset "MatrixShape equality and hashing are by value" begin
    a = KI.MatrixShape(Float16, Float32, 16, 16, 16, KI.SubgroupScope())
    b = KI.MatrixShape(Float16, Float32, 16, 16, 16, KI.SubgroupScope())
    @test a == b
    @test hash(a) == hash(b)
    @test length(Set([a, b])) == 1
    @test a != KI.MatrixShape(Float16, Float16, 16, 16, 16, KI.SubgroupScope())   # acc differs
    @test a != KI.MatrixShape(Float16, Float32, 16, 16, 16, KI.WorkgroupScope())  # scope differs

    # `show` names the type pair, the extents and the scope, in that order.
    @test sprint(show, a) == "MatrixShape(Float16->Float32 16x16x16 SubgroupScope)"
end

@testset "supports matches on type, not just extents" begin
    @test KI.supports(THISCARD, KI.MatrixShape(Float16, Float32, 16, 16, 16, KI.SubgroupScope()))
    # Same extents, a type combination the card does not list. Matching on
    # (M,N,K) alone would say yes and emit instructions the device cannot run.
    @test !KI.supports(THISCARD, KI.MatrixShape(Float32, Float32, 16, 16, 16, KI.SubgroupScope()))
    @test !KI.supports(KI.MatrixShape[], KI.MatrixShape(Float16, Float32, 16, 16, 16, KI.SubgroupScope()))
end

@testset "bestshape" begin
    @test KI.bestshape(THISCARD, Float16, Float32).M == 16
    @test KI.bestshape(THISCARD, Int8, Int32).acc === Int32
    # No matrix hardware, or no shape for this type pair: `nothing`, never a
    # fabricated tile — the value is used as a divisor.
    @test KI.bestshape(KI.MatrixShape[], Float16, Float32) === nothing
    @test KI.bestshape(THISCARD, Float64, Float64) === nothing
    @test KI.bestshape(THISCARD, Float16, Float32; scope = KI.WorkgroupScope()) === nothing
end

@testset "bestshape prefers square, then large" begin
    mixed = [
        KI.MatrixShape(Float16, Float32, 16, 8, 32, KI.SubgroupScope()),   # skewed
        KI.MatrixShape(Float16, Float32, 8, 8, 8, KI.SubgroupScope()),     # square, small
        KI.MatrixShape(Float16, Float32, 16, 16, 16, KI.SubgroupScope()),  # square, large
    ]
    @test KI.bestshape(mixed, Float16, Float32) == mixed[3]
    # With no square option it still returns the least skewed rather than nothing.
    skewed = [
        KI.MatrixShape(Float16, Float32, 16, 8, 32, KI.SubgroupScope()),
        KI.MatrixShape(Float16, Float32, 16, 8, 16, KI.SubgroupScope()),
    ]
    @test KI.bestshape(skewed, Float16, Float32) == skewed[2]
end

@testset "DeviceCaps" begin
    c = KI.caps(MatrixBackend())

    # Eight positional arguments still construct one, and they mean "no
    # workgroup-scope matrices" plus "a square `tile` fp16 -> fp32 instruction" —
    # so `tile` and `shapes` describe the same device rather than disagreeing.
    eight = KI.DeviceCaps(true, 8, 32, 32, 32768, 1024, 20, 64)
    @test isempty(eight.wggran)
    @test eight.shapes == [KI.MatrixShape(Float16, Float32, 8, 8, 8, KI.SubgroupScope())]
    @test isempty(KI.DeviceCaps(false, 0, 32, 32, 32768, 1024, 20, 64).shapes)

    # A copy changes exactly the fields it names. `coopmat = false` in
    # particular does NOT empty the table: naming one field and moving three is
    # how a copy stops meaning what it says.
    off = KI.DeviceCaps(c; coopmat = false)
    @test off.coopmat === false
    @test off.shapes === c.shapes          # untouched…
    @test off.wggran === c.wggran
    @test isempty(KI.matrix_shapes(off))   # …and the accessor still answers as
                                           #    the device it claims to be
    @test KI.DeviceCaps(c; subgroup = 64).subgroup == 64
    @test KI.DeviceCaps(c; subgroup = 64).coopmatsubgroup == c.coopmatsubgroup

    # Multiples coarsen as the workgroup grows, and a size with no row is
    # `nothing` rather than a fabricated tile.
    @test KI.wggranularity(c, 32) == (16, 16, 16)
    @test KI.wggranularity(c, 128) == (32, 16, 16)
    @test KI.wggranularity(c, 256) == (32, 32, 16)
    @test KI.wggranularity(c, 64) === nothing
    @test KI.wggranularity(KI.DeviceCaps(true, 16, 32, 32, 0, 1024, 0, 0), 32) === nothing
end

@testset "shape queries derive from caps" begin
    fp16 = KI.MatrixShape(Float16, Float32, 16, 16, 16, KI.SubgroupScope())
    fp32 = KI.MatrixShape(Float32, Float32, 16, 16, 16, KI.SubgroupScope())

    # A backend answers `caps` and gets every one of these for free.
    @test KI.matrix_shapes(MatrixBackend()) == THISCARD
    @test KI.supports(MatrixBackend(), fp16)
    @test !KI.supports(MatrixBackend(), fp32)
    @test KI.bestshape(MatrixBackend(), Int8, Int32).acc === Int32
    @test KI.bestshape(MatrixBackend(), Float16, Float32; scope = KI.WorkgroupScope()) === nothing

    # With cooperative matrices off, every one of them says so — through the
    # backend and through the caps, with the same answer.
    @test KI.matrix_shapes(NoMatrixBackend()) == KI.MatrixShape[]
    @test !KI.supports(NoMatrixBackend(), fp16)
    @test KI.bestshape(NoMatrixBackend(), Float16, Float32) === nothing
    off = KI.caps(NoMatrixBackend())
    @test !KI.supports(off, fp16)
    @test KI.bestshape(off, Float16, Float32) === nothing

    # `caps` is required of a backend, not defaulted: one that has not answered
    # it gets a MethodError rather than a plausible empty device.
    @test_throws MethodError KI.caps(StubBackend())
    @test_throws MethodError KI.matrix_shapes(StubBackend())
end

@testset "allocate / zeros / ones" begin
    b = AllocBackend()

    # Dims given as varargs are forwarded to the tuple method backends implement.
    @test KI.allocate(b, Float32, (2,)) isa Vector{Float32}
    @test size(KI.allocate(b, Float32, 2, 3)) == (2, 3)

    @test KI.zeros(b, Float64, 2, 3) == zeros(2, 3)
    @test KI.ones(b, Int, (4,)) == ones(Int, 4)

    # A backend without `allocate` yields a MethodError pointing at the missing
    # method — including via the keyword form — and a clear error when unified
    # memory is requested but not supported.
    @test_throws MethodError KI.allocate(StubBackend(), Float32, (2,))
    @test_throws MethodError KI.allocate(StubBackend(), Float32, (2,); unified = false)
    @test_throws ArgumentError KI.allocate(StubBackend(), Float32, (2,); unified = true)
end

@testset "_print" begin
    # The host fallback keeps `KernelAbstractions.@print` working outside a kernel.
    # `@print` wraps literals in `Val` so backends can use them as format strings;
    # the fallback has to unwrap them again.
    @test capture_stdout(() -> KI._print()) == ""
    @test capture_stdout(() -> KI._print(Val(Symbol("hello\n")))) == "hello\n"
    @test capture_stdout(() -> KI._print(1, 2)) == "12"
    @test capture_stdout(() -> KI._print(Val(Symbol("x = ")), 42, Val(Symbol("\n")))) ==
        "x = 42\n"
    @test capture_stdout(() -> KI._print(Val(3), " ", Val(:sym))) == "3 sym"
end

@testset "check_launch_args" begin
    # Validation only: valid configurations pass through without normalization.
    @test KI.check_launch_args(1, 1, ()) === nothing
    @test KI.check_launch_args((1, 2, 3), (1, 2, 3), ()) === nothing
    @test KI.check_launch_args((), 1, 1) === nothing
    @test KI.check_launch_args((), (), ()) === nothing

    @test_throws ArgumentError KI.check_launch_args((1, 2, 3, 4), 1, ())
    @test_throws ArgumentError KI.check_launch_args(1, (1, 2, 3, 4), ())
    @test_throws ArgumentError KI.check_launch_args((), 1, (1, 2, 3, 4))
    @test_throws ArgumentError KI.check_launch_args(2, 4, 2) # both numworkgroupsize and ndrange defined
    @test_throws ArgumentError KI.check_launch_args(2, (), 2) # both numworkgroupsize and ndrange defined
end

@testset "threads_to_workgroupsize" begin
    # Fills dimensions left to right without exceeding the thread budget.
    @test KI.threads_to_workgroupsize(256, (1000,)) == (256,)
    @test KI.threads_to_workgroupsize(256, (100,)) == (100,)
    @test KI.threads_to_workgroupsize(256, (100, 50)) == (100, 2)
    @test KI.threads_to_workgroupsize(1024, (5, 5, 5)) == (5, 5, 5)
    @test KI.threads_to_workgroupsize(4, (3, 3)) == (3, 1)
    @test prod(KI.threads_to_workgroupsize(256, (100, 50))) <= 256

    # Zero-sized dimensions are clamped to 1 so the launch math stays defined.
    @test KI.threads_to_workgroupsize(256, (0, 4)) == (1, 4)
    @test KI.threads_to_workgroupsize(256, (4, 0)) == (4, 1)
    @test KI.threads_to_workgroupsize(0, (5,)) == (1,)
end

@testset "Kernel" begin
    kernel = KI.Kernel(:backend, :kern)
    @test kernel.backend === :backend
    @test kernel.kern === :kern
end

# A backend reporting a fixed workgroup-size limit, for exercising the
# auto-sizing helper.
struct SizedBackend <: KI.Backend
    maxThreads::Int
end
function KI.kernel_max_work_group_size(k::KI.Kernel{SizedBackend}; max_work_items::Int = typemax(Int))
    return min(k.backend.maxThreads, max_work_items)
end

@testset "auto_launch_sizes" begin
    kernel = KI.Kernel(SizedBackend(256), nothing)

    # Without an ndrange the sizes pass through, defaulting to 1.
    @test KI.auto_launch_sizes(kernel, (), (), ()) === (1, 1)
    @test KI.auto_launch_sizes(kernel, 4, (), ()) === (4, 1)
    @test KI.auto_launch_sizes(kernel, (), (2, 2), ()) === (1, (2, 2))
    @test KI.auto_launch_sizes(kernel, (4, 4), (2, 2), ()) === ((4, 4), (2, 2))

    # With an ndrange and no workgroupsize, the workgroupsize is derived from
    # the kernel's limit and the workgroup count covers the ndrange.
    @test KI.auto_launch_sizes(kernel, (), (), (1000,)) === ((4,), (256,))
    @test KI.auto_launch_sizes(kernel, (), (), (10,)) === ((1,), (10,))
    @test KI.auto_launch_sizes(kernel, (), (), (100, 50)) === ((1, 25), (100, 2))
    @test KI.auto_launch_sizes(kernel, (), (), 1000) === (4, 256)

    # An explicit workgroupsize is kept as-is.
    @test KI.auto_launch_sizes(kernel, (), (16,), (100,)) === ((7,), (16,))

    # A zero-sized ndrange yields zero workgroups; backends skip the launch.
    @test KI.auto_launch_sizes(kernel, (), (), (0,)) === ((0,), (1,))
    @test KI.auto_launch_sizes(kernel, (), (), (0, 4)) === ((0, 4), (1, 1))
    @test KI.auto_launch_sizes(kernel, (), (), 0) === (0, 1)
end

@testset "split_kwargs" begin
    kwargs = [:(launch = false), :(name = "foo"), :(numworkgroups = 2)]
    macro_kw, compiler_kw, launch_kw, other = KI.split_kwargs(
        kwargs, KI.MACRO_KWARGS, KI.COMPILER_KWARGS, KI.LAUNCH_KWARGS
    )
    @test macro_kw == [:(launch = false)]
    @test compiler_kw == [:(name = "foo")]
    @test launch_kw == [:(numworkgroups = 2)]
    @test isempty(other)

    # Unmatched keywords land in the trailing group rather than erroring.
    _, unmatched = KI.split_kwargs([:(bogus = 1)], [:launch])
    @test unmatched == [:(bogus = 1)]

    # Also usable at run time with pairs instead of expressions.
    matched, _ = KI.split_kwargs([:launch => false], [:launch])
    @test matched == [:launch => false]

    @test_throws ArgumentError KI.split_kwargs([:(f(x))], [:launch])
    @test_throws ArgumentError KI.split_kwargs([Expr(:(=), 1, 2)], [:launch])
end

@testset "assign_args!" begin
    code = Expr(:block)
    vars, var_exprs = KI.assign_args!(code, [:a, :(b...)])
    @test length(vars) == 2
    # Arguments are hoisted into gensyms so the caller can `GC.@preserve` them.
    @test code.args == [:($(vars[1]) = a), :($(vars[2]) = b)]
    @test var_exprs[1] === vars[1]
    @test var_exprs[2] == Expr(:..., vars[2])
end

# A minimal backend, exercising the contract `KI.@kernel` expects of one.
struct MockBackend end

struct MockKernel
    f::Any
    tt::Any
    name::Any
    launches::Vector{Any}
end

KI.argconvert(::MockBackend, arg) = arg
function KI.kernel_function(::MockBackend, f, tt = Tuple{}; name = nothing, kwargs...)
    return MockKernel(f, tt, name, [])
end
function (kernel::MockKernel)(args...; kwargs...)
    push!(kernel.launches, (args, Dict(kwargs)))
    return nothing
end

dummy(a, b) = nothing

@testset "@kernel" begin
    backend = MockBackend()

    kernel = KI.@kernel backend numworkgroups = 2 workgroupsize = 4 dummy(1, 2.0)
    @test kernel isa MockKernel
    @test kernel.f === dummy
    @test kernel.tt == Tuple{Int, Float64}
    args, launch_kwargs = only(kernel.launches)
    @test args == (1, 2.0)
    @test launch_kwargs == Dict(:numworkgroups => 2, :workgroupsize => 4)

    # `launch=false` compiles only; the caller launches later.
    deferred = KI.@kernel backend launch = false dummy(1, 2.0)
    @test isempty(deferred.launches)

    # Compiler kwargs reach `kernel_function` instead of the launch.
    named = KI.@kernel backend launch = false name = "mykernel" dummy(1, 2.0)
    @test named.name == "mykernel"

    # Splatted arguments are supported.
    splatted = KI.@kernel backend launch = false dummy((1, 2.0)...)
    @test splatted.tt == Tuple{Int, Float64}

    @testset "errors" begin
        # These throw during macro expansion, so they cannot be written as a plain
        # `@test_throws` call. `macroexpand` wraps such errors in a `LoadError`.
        function expansion_error(ex)
            try
                macroexpand(@__MODULE__, ex)
            catch err
                return err isa LoadError ? err.error : err
            end
            return nothing
        end

        @test expansion_error(:(KI.@kernel backend bogus = 1 dummy(1))) isa ArgumentError
        @test expansion_error(:(KI.@kernel backend dummy)) isa ArgumentError
        @test expansion_error(:(KI.@kernel backend launch = 1 dummy(1))) isa ArgumentError
        @test expansion_error(:(KI.@kernel backend "notakwarg" dummy(1))) isa ArgumentError
        # launch-time kwargs are meaningless when we are not launching
        @test expansion_error(
            :(KI.@kernel backend launch = false numworkgroups = 2 dummy(1))
        ) isa ErrorException
    end
end

# The mesh pipeline's emitter. All of this is portable arithmetic with no driver
# in it, which is the point: the geometry-to-mesh lowering is only worth more
# than a per-backend rewrite if it can be RUN and checked without a GPU.
@testset "MeshEmitter" begin
    cull = (position = 0.0f0,)
    verts(n) = [(position = Float32(i),) for i in 1:n]

    @testset "MeshConfig" begin
        c = KI.MeshConfig(max_vertices = 4, max_primitives = 2,
                          topology = KI.TriangleStrip(), threads = 32)
        @test c.max_vertices == 4
        @test c.max_primitives == 2
        @test c.threads == 32
        # The topology is a type parameter because a compiler dispatches on it.
        @test c isa KI.MeshConfig{KI.TriangleStrip}
        @test_throws ArgumentError KI.MeshConfig(max_vertices = 0, max_primitives = 1)
        @test_throws ArgumentError KI.MeshConfig(max_vertices = 1, max_primitives = 0)
        @test_throws ArgumentError KI.MeshConfig(max_vertices = 1, max_primitives = 1,
                                                 threads = 0)
    end

    @testset "topologies" begin
        function run(T, n; maxprim = 8)
            out = KI.HostMeshOutput()
            e = KI.MeshEmitter{T}(out, 1, 1, maxprim)
            for v in verts(n)
                KI.emit!(e, v)
            end
            return out, e
        end

        @test KI.primitives(first(run(KI.TriangleList, 6))) == [(1, 2, 3), (4, 5, 6)]
        @test KI.primitives(first(run(KI.TriangleStrip, 4))) == [(1, 2, 3), (3, 2, 4)]
        @test KI.primitives(first(run(KI.LineStrip, 4))) == [(1, 2), (2, 3), (3, 4)]
        @test KI.primitives(first(run(KI.LineList, 4))) == [(1, 2), (3, 4)]
        @test KI.primitives(first(run(KI.PointList, 3))) == [(1,), (2,), (3,)]

        # A vertex is written for every emit!, whatever the topology does with it.
        for (T, n) in ((KI.TriangleList, 6), (KI.TriangleStrip, 4),
                       (KI.LineStrip, 4), (KI.LineList, 4), (KI.PointList, 3))
            out, e = run(T, n)
            @test sort!(collect(keys(out.vertices))) == Int32.(1:n)
            @test KI.nvertices(e) == n
        end
    end

    # The reason a strip alternates winding is ORIENTATION, so that is what gets
    # measured. Reading the index tuples back would pass just as happily on the
    # rule that makes every second triangle back-facing, which `CullBack` then
    # drops: half a quad, and it reads as a shader bug.
    @testset "strip winding" begin
        quad = [(0.0f0, 0.0f0), (1.0f0, 0.0f0), (0.0f0, 1.0f0), (1.0f0, 1.0f0)]
        area(a, b, c) = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])

        out = KI.HostMeshOutput()
        e = KI.MeshEmitter{KI.TriangleStrip}(out, 1, 1, 2)
        for c in 1:4
            KI.emit!(e, (position = quad[c],))
        end
        signs = map(KI.primitives(out)) do (i, j, k)
            area(out.vertices[i].position, out.vertices[j].position,
                 out.vertices[k].position)
        end
        @test length(signs) == 2
        @test all(>(0), signs) || all(<(0), signs)
        # Without the alternation the second triangle would be (2,3,4), and this
        # is the assertion that tells the two rules apart.
        @test sign(area(quad[2], quad[3], quad[4])) != sign(signs[1])
    end

    @testset "endprimitive! restarts the run" begin
        out = KI.HostMeshOutput()
        e = KI.MeshEmitter{KI.TriangleStrip}(out, 1, 1, 8)
        for c in 1:3
            KI.emit!(e, (position = Float32(c),))
        end
        KI.endprimitive!(e)
        for c in 4:6
            KI.emit!(e, (position = Float32(c),))
        end
        # No triangle spans the seam: (2,3,4) and (3,4,5) would both be wrong.
        @test KI.primitives(out) == [(1, 2, 3), (4, 5, 6)]
    end

    @testset "finish! fills the reserved slots" begin
        # An invocation that emits nothing at all, which is what a culled
        # zero-width line does. Its slots are still in the threadgroup's count.
        out = KI.HostMeshOutput()
        e = KI.MeshEmitter{KI.TriangleStrip}(out, 1, 1, 2)
        KI.finish!(e, cull)
        @test KI.nprimitives(e) == 2
        @test KI.primitives(out) == [(1, 1, 1), (1, 1, 1)]
        # The cull vertex is written, so the degenerate indices point at a
        # position rather than at whatever the slot happened to hold.
        @test out.vertices[Int32(1)] === cull

        # Partially filled: three vertices close one triangle and leave one slot.
        out = KI.HostMeshOutput()
        e = KI.MeshEmitter{KI.TriangleStrip}(out, 1, 1, 2)
        for c in 1:3
            KI.emit!(e, (position = Float32(c),))
        end
        KI.finish!(e, cull)
        @test KI.primitives(out) == [(1, 2, 3), (4, 4, 4)]
        @test out.vertices[Int32(4)] === cull

        # Filled to the budget: nothing to do, and no cull vertex written.
        out = KI.HostMeshOutput()
        e = KI.MeshEmitter{KI.TriangleStrip}(out, 1, 1, 2)
        for c in 1:4
            KI.emit!(e, (position = Float32(c),))
        end
        KI.finish!(e, cull)
        @test length(out.vertices) == 4
        @test KI.primitives(out) == [(1, 2, 3), (3, 2, 4)]
    end

    # A fixed range per invocation rather than a shared cursor is what keeps the
    # geometry stage's order guarantee: input primitive order is invocation
    # order is slot order, with no synchronisation to get there.
    @testset "cooperating invocations" begin
        out = KI.HostMeshOutput()
        for t in 1:4
            e = KI.MeshEmitter{KI.TriangleStrip}(out, (t - 1) * 4 + 1, (t - 1) * 2 + 1, 2)
            for c in 1:4
                KI.emit!(e, (position = (Float32(t), Float32(c)),))
            end
            KI.endprimitive!(e)
            KI.finish!(e, (position = (0.0f0, 0.0f0),))
        end
        KI.set_mesh_outputs!(out, 16, 8)

        @test sort!(collect(keys(out.vertices))) == Int32.(1:16)
        @test KI.primitives(out) == [(1, 2, 3), (3, 2, 4), (5, 6, 7), (7, 6, 8),
                                     (9, 10, 11), (11, 10, 12), (13, 14, 15), (15, 14, 16)]
        @test out.declared[] == (Int32(16), Int32(8))
    end

    @testset "host answers" begin
        # These say what went wrong rather than leaving a bare MethodError on a
        # zero-argument function to be read as a missing method.
        @test_throws ErrorException KI.mesh_thread_index()
        @test_throws ErrorException KI.mesh_group_index()
        @test_throws ErrorException KI.emit!(KI.NativeEmitter(), (position = 0.0f0,))
        @test_throws ErrorException KI.endprimitive!(KI.NativeEmitter())
        # The writers dispatch on the output object, so a type with no methods
        # is a MethodError naming it, which is the accurate complaint.
        @test_throws MethodError KI.set_mesh_vertex!(nothing, 1, (position = 0.0f0,))
        @test_throws MethodError KI.set_mesh_triangle!(nothing, 1, 1, 2, 3)
        @test_throws MethodError KI.set_mesh_outputs!(nothing, 1, 1)
    end
end

@testset "Aqua" begin
    Aqua.test_all(KernelInterface)
end
