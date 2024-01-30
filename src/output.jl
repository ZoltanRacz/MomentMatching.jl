#= This file produceses figures and tables to display results from estimation.
=#

"""
$(TYPEDSIGNATURES)

Plots objective function around mimimizer.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
- margobj: Array with objective function values around the minimizer.

# Optional arguments
- glob: Logical, true if only global stage was performed.
- saving: Logical, true if output has to be saved.
- plotarg: Other inputs for plotting.
"""
function fmarg(estset::EstimationSetup, mmsolu::EstimationResult, margobj::AbstractArray; glob::Bool=mmsolu.npmm.onlyglo, saving::Bool=false, filename_suffix::String="", plotarg...)
    @unpack xglo, xloc, fglo, floc,npmm = mmsolu

    labs = labels(estset)

    glob ? x = xglo[1] : x = xloc[1]
    glob ? obj = fglo[1] : obj = floc[1]

    fig = Any[]

    for k in axes(margobj, 2)

        f = plot(margobj[:, k, 1], margobj[:, k, 2]; title=labs[k], label="",plotarg...)


        xopt = x[k]
        fopt = obj
        scatter!([xopt], [fopt];
            label="",
            markercolor="red",
            markersize=3,
            markershape=:circle,
            markeralpha=1.0,
            markerstrokewidth=1, plotarg...)
        push!(fig, f)
    end
    ffig = plot(fig...;fonts()...)

    saving && savefig(ffig, estimation_output_path() * estimation_name(estset, npmm, filename_suffix) * "_fmarg.pdf")

    display(ffig)

    return ffig
end

"""
$(TYPEDSIGNATURES)

Plots the series of estimated parameters in all trials.

If solution is stable around minimizer then this indicates stability.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).

# Optional arguments
- glob: Logical, true if only global stage was performed.
- saving: Logical, true if output has to be saved.
- firstN: How many observations to plot.
- plotarg: Other inputs for plotting.
"""
function fsanity(estset::EstimationSetup, mmsolu::EstimationResult; glob::Bool = mmsolu.npmm.onlyglo, saving::Bool=false, firstN::Integer = 1000, filename_suffix::String="", ylimss = fill(:none,1+length(mmsolu.xglo[1]))::AbstractVector)
    @unpack mode, modelname = estset
    @unpack fglo, floc, xglo, xloc, conv, npmm = mmsolu

    glob ? ob = fglo : ob = floc
    glob ? x = xglo : x = xloc

    labs = labels(estset)
    lastindex = min(firstN,length(ob))

    fig = Any[]

    f = plot(ob[1:lastindex]; title="objective", label="", titlefont=font(10), ylims = ylimss[1])
    !glob && scatter!(collect(1:lastindex)[conv[1:lastindex]],ob[1:lastindex][conv[1:lastindex]]; label="converged", color=:green, markersize=2.0)
    !glob && scatter!(collect(1:lastindex)[.!(conv[1:lastindex])],ob[1:lastindex][.!(conv[1:lastindex])]; label="not converged", color=:red, markersize=2.0)
    push!(fig, f)

    xx = hcat([x[j] for j in 1:length(ob)]...)

    for k in axes(xx,1)
        f = plot(xx[k, 1:lastindex], title=labs[k]; label="", ylims = ylimss[k+1])
        plot!(ones(lastindex) * xx[k, 1]; label="", color=:red, linestyle=:dash)
        push!(fig, f)
    end

    ffig = plot(fig...; fonts()...,legendfontsize=4)

    saving && savefig(ffig, estimation_output_path() * estimation_name(estset, npmm, filename_suffix) * "_fsanity.pdf")

    display(ffig)

    return ffig
end

function fglobounds(estset::EstimationSetup, mmsolu::EstimationResult; saving::Bool=false, filename_suffix::String="")
    @unpack mode, modelname = estset
    @unpack xglo, npmm = mmsolu


    labs = labels(estset)
    
    fig = Any[]

    
    xx = hcat(xglo...)

    N = 10
    percs = [1/5^(i-1) for i in 1:N]

    for k in axes(xx,1)
        mins = [minimum(@view xx[k,1:floor(Int,npmm.Nglo*percs[i])]) for i in 1:N]
        maxs = [maximum(@view xx[k,1:floor(Int,npmm.Nglo*percs[i])]) for i in 1:N]
        f = plot(percs,mins, title=labs[k]; label="", color=:blue, xaxis=:log10)
        plot!(percs,maxs; label="", color=:blue, xaxis=:log10)
        push!(fig, f)
    end

    ffig = plot(fig...; fonts()...,legendfontsize=4)

    saving && savefig(ffig, estimation_output_path() * estimation_name(estset, npmm, filename_suffix) * "_fglobounds.pdf")

    display(ffig)

    return ffig
end


"""
$(TYPEDSIGNATURES)

Creates DataFrame with parameter estimates.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).

# Optional arguments
- glob: Logical, true if only global stage was performed.
"""
function tableest_inner(estset::EstimationSetup,mmsolu::EstimationResult; glob::Bool=mmsolu.npmm.onlyglo)
    @unpack xglo, xloc,npmm = mmsolu

    labs = labels(estset)
    glob ? x = xglo[1] : x = xloc[1]

    return DataFrame(:Variable=>labs, Symbol("Point estimate")=>round.(x, digits=3))
end

"""
$(TYPEDSIGNATURES)

Creates DataFrame with parameter estimates and optionally saves it.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).

# Optional arguments
- glob: Logical, true if only global stage was performed.
- saving: Logical, true if output has to be saved.
"""
function tableest(estset::EstimationSetup,mmsolu::EstimationResult; glob::Bool=mmsolu.npmm.onlyglo, saving::Bool=false, filename_suffix::String="")
    @unpack mode, modelname = estset
    @unpack npmm = mmsolu

    df = tableest_inner(estset,mmsolu; glob)

    saving && CSV.write(estimation_output_path() * estimation_name(estset, npmm, filename_suffix) * "_tableest.csv", df)

    println(df)

    return df
end

"""
$(TYPEDSIGNATURES)

Creates DataFrame with parameter estimates and optionally saves it, bootstrap case.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
- boot: Instance of BootstrapResult. See separate documentation [`BootstrapResult`](@ref).

# Optional arguments
- cilev: Level of confidence intervals.
- glob: Logical, true if only global stage was performed.
- saving: Logical, true if output has to be saved.
"""
function tableest(estset::EstimationSetup,mmsolu::EstimationResult, boot::BootstrapResult; cilev::Real=0.05, glob::Bool=mmsolu.npmm.onlyglo, saving::Bool=false, filename_suffix::String="")
    @unpack mode, modelname = estset
    @unpack npmm = mmsolu
    @unpack xs,sd_asymp = boot

    df = tableest_inner(estset,mmsolu; glob)

    bs_sds = [sqrt(var(xs[i,:,:])) for i in axes(xs,1)]
    ratios = [mean(var(xs[i,:,:]; dims=1))/var(xs[i,:,:]) for i in axes(xs,1)]
    bs_ci = [quantile(xs[i,:,:][:], [cilev/2, 1.0 - cilev/2]) for i in axes(xs, 1)]

    df[:, :("Asymptotic standard errors")] = round.(sd_asymp, digits=3)
    df[:, :("Bootstrapped standard errors")] = round.(bs_sds, digits=3)
    df[:, :("Bootstrapped $(1-cilev)% CI")] = round.(bs_ci, digits=3)
    df[:, :("Seed share of variance")] = round.(ratios, digits=3)

    saving && CSV.write(estimation_output_path() * estimation_name(estset, npmm, filename_suffix) * "_tableest_boot.csv", df)

    println(df)

    return df
end

"""
$(TYPEDSIGNATURES)

Compares model-generated moments with data in a table.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
"""
function tablemoms_inner(estset::EstimationSetup, mmsolu::EstimationResult)
    @unpack mode, typemom = estset
    @unpack momloc, npmm = mmsolu

    momsdata = datamoments(mode, typemom) # moment from data to match

    df = momentnames(mode, typemom)

    df[:, :("Sample values")] = round.(momsdata[:,1], digits=3)
    df[:, :("Model values")] = round.(momloc[1], digits=3)

    return df
end

"""
$(TYPEDSIGNATURES)

Compares model-generated moments with data in a table and optionally saves output.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).

# Optional arguments
- saving: Logical, true if output has to be saved.
"""
function tablemoms(estset::EstimationSetup, mmsolu::EstimationResult; saving::Bool=false, filename_suffix::String="")
    @unpack mode, modelname = estset
    @unpack npmm = mmsolu

    df = tablemoms_inner(estset, mmsolu)

    saving && CSV.write(estimation_output_path() * estimation_name(estset, npmm, filename_suffix) * "_tablemoms.csv", df)

    println(df)

    return df
end

"""
$(TYPEDSIGNATURES)

Compares model-generated moments with data in a table and optionally saves output, bootstrap case.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
- boot: Instance of BootstrapResult. See separate documentation [`BootstrapResult`](@ref).

# Optional arguments
- saving: Logical, true if output has to be saved.
"""
function tablemoms(estset::EstimationSetup, mmsolu::EstimationResult, boot::BootstrapResult; saving::Bool=false, filename_suffix::String="")
    @unpack mode, modelname = estset
    @unpack npmm = mmsolu
    @unpack moms, W = boot

    df = tablemoms_inner(estset, mmsolu)

    df[:, :("Simulated standard errors")] = round.(sqrt.(diag(Omega_boots(moms))), digits=3)
    df[:, :("Efficient weights (approx)")] = round.(diag(W), digits=3)

    saving && CSV.write(estimation_output_path() * estimation_name(estset, npmm, filename_suffix) * "_tablemoms_boot.csv", df)

    println(df)

    return df
end

"""
$(TYPEDSIGNATURES)

compares model-generated moments with data in a plot
"""
function fmoms(estset::EstimationSetup, mmsolu::EstimationResult; saving::Bool=false, filename_suffix::String="")
    @unpack momloc, npmm = mmsolu

    df = tablemoms_inner(estset, mmsolu)

    if ncol(df) == 3
        insertcols!(df, 1, :empty => "Moments")
    end

    titles = unique(df[:,1])
    xlb = names(df)[2]

    figs = Array{Plots.Plot{Plots.GRBackend},1}(undef, length(titles)) # initialize array of figures

    # plot attributes
    lc = [palette(:tab10)[1], palette(:tab10)[4]]
    ls = [:solid, :dash]
    msh = [:circle, :x]
    msz = [5, 6]
    lb = ["Data", "Model"]
#    yls = [(0.3, 1.0), (-0.3, 0.3), (0.3, 1.0), (-0.3, 0.3)]

    for ff in eachindex(titles)
        figs[ff] = plot()
        for k in eachindex(lb)
            lb1 = lb[k]
            if ff != 1
                lb1 = ""
            end
            plot!(df[df[:,1] .== titles[ff], 2], df[df[:,1] .== titles[ff], 3:4][:,k],
                label=lb1,
                linecolor=lc[k],
                linestyle=ls[k],
                linewidth=1.5,
                markercolor=lc[k],
                markersize=msz[k],
                markershape=msh[k],
                markeralpha=1.0,
                markerstrokecolor=lc[k],
                markerstrokewidth=2)
        end
        figs[ff] = plot(figs[ff], title=titles[ff],
            legend=:topleft,
            size=(650, 540), 
            #ylims=yls[ff],
            foreground_color_legend=nothing,
            xlabel=xlb,
            xrotation=45,
            titlefont=font(14),
            titlelocation=:left,
            xguidefont=font(12),
            yguidefont=font(12),
            tickfont=font(10),
            legendfont=font(10),
            left_margin=5Plots.PlotMeasures.mm,
            right_margin=5Plots.PlotMeasures.mm,
            top_margin=2.5Plots.PlotMeasures.mm,
            bottom_margin=5Plots.PlotMeasures.mm)

            display(figs[ff])

        saving && savefig(figs[ff], estimation_output_path() * estimation_name(estset, npmm, filename_suffix) * titles_s[ff] * "_fmoms.pdf")
    end
    return figs
end

"""
$(TYPEDSIGNATURES)

Plot bootstrap results.

# Required arguments
- estset: Instance of EstimationSetup. See separate documentation [`EstimationSetup`](@ref).
- mmsolu: Instance of EstimationResult. See separate documentation [`EstimationResult`](@ref).
- xs: Array with bootstrap values.

# Optional arguments
- ci: Logical, true if confidence intervals should be plotted.
- cilev: Level of confidence intervals.
- saving: Logical, true if output has to be saved.
- plotarg: Other inputs for plotting.
"""
function fbootstrap(estset::EstimationSetup, mmsolu::EstimationResult, boot::BootstrapResult; ci::Bool=true, cilev::Real=0.05, saving::Bool=false, filename_suffix::String="", trim::Real = 0.01)
    @unpack npmm,xloc = mmsolu
    @unpack xs = boot

    labs = labels(estset)

    fig = Any[]

    for k in axes(xs, 1)
        xdist = sort(xs[k,:,:][:])
        toskip = floor(Int,length(xdist)*trim)
        xdist_trimmed = xdist[(1+toskip):(end-toskip)]

        f = histogram(xdist_trimmed; title=labs[k], label="")


        xopt = xloc[1][k]
        vline!([xopt];
            label="",
            linecolor="red",
            width=2,
            style=:dash)
        if ci==true
            vline!(quantile(xdist, [cilev/2, 1.0 - cilev/2], sorted=true);
                label="",
                linecolor="red",
                width=2,
                style=:dot)
        end
        push!(fig, f)
    end
    ffig = plot(fig...;fonts()...)

    saving && savefig(ffig, estimation_output_path() * estimation_name(estset, npmm, filename_suffix) * "_fboot.pdf")

    display(ffig)

    return ffig
end

"""
$(TYPEDSIGNATURES)

Some plot attributes often used.
"""
function fonts()
    return Dict( 
    :xrotation=>45,
    :titlefont=>font(8),
    :titlelocation=>:left,
    :xguidefont=>font(6),
    :yguidefont=>font(6),
    :tickfont=>font(6),
    :legendfont=>font(6)
    )
end

"""
$(TYPEDSIGNATURES)

Produce all figures and tables
"""
function alloutputs(estset::EstimationSetup, mmsolu::EstimationResult, boot::BootstrapResult; saving::Bool = true, filename_suffix::String="")
    marg = marginal_fobj(estset, mmsolu)
    fmarg(estset, mmsolu,marg; saving, filename_suffix)
    fsanity(estset, mmsolu; saving, filename_suffix)
    tableest(estset, mmsolu, boot; saving, filename_suffix) 
    fmoms(estset, mmsolu; saving, filename_suffix) 
    tablemoms(estset, mmsolu, boot; saving, filename_suffix) 
    fbootstrap(estset, mmsolu, boot; saving, filename_suffix)
    return nothing
end
