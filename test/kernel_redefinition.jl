module KernelRedefinition
using KernelAbstractions, Test

function kernel_redefinition_testsuite(backend=CPU())
    mod = Module(gensym(:ReloadedKernel))
    Core.eval(mod,:(using KernelAbstractions))
    Core.eval(mod,quote
        @kernel function fill_value!(out)
            i=@index(Global)
            out[i]=Int32(1)
        end
    end)
    kernel_function = Base.invokelatest(getproperty,mod,:fill_value!)
    original = Set(methods(kernel_function))
    # A second kernel overload must retain its original constructors, rather
    # than redefine methods while a package is being precompiled.
    Core.eval(mod,quote
        @kernel function fill_value!(out,value::Int32)
            i=@index(Global)
            out[i]=value
        end
    end)
    @test Set(methods(kernel_function))==original

    # Revise deletes a macro expansion's methods before reevaluating it. Julia
    # keeps the function binding, which used to suppress constructor recovery.
    foreach(Base.delete_method,collect(methods(kernel_function)))
    @test isdefined(mod,:fill_value!)
    @test isempty(methods(kernel_function))
    Core.eval(mod,quote
        @kernel function fill_value!(out)
            i=@index(Global)
            out[i]=Int32(2)
        end
    end)
    @test Base.invokelatest(hasmethod,kernel_function,Tuple{Any})
    @test length(methods(kernel_function))==length(original)
    output=KernelAbstractions.zeros(backend,Int32,8)
    kernel=Base.invokelatest(kernel_function,backend,32)
    Base.invokelatest(kernel,output;ndrange=8)
    synchronize(backend)
    @test Array(output)==fill(Int32(2),8)
end
end
