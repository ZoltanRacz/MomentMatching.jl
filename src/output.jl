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
- plotarg: Other inputs for plotting.
"""
fmarg

@userplot FMarg

@recipe function f(h::FMarg; glob=h.args[2].npmm.onlyglo, which_point = 1)
    if length(h.args) != 3 || !(h.args[1] isa EstimationSetup) ||
       !(h.args[2] isa EstimationResult) || !(h.args[3] isa AbstractArray)
        error("fmarg should be given three inputs: an EstimationSetup, an EstimationResult and an AbstractArray. Got: $(typeof(h.args))")
    end
    estset, mmsolu, margobj = h.args
    @unpack xglo, xloc, fglo, floc, npmm = mmsolu

    labs = labels(estset)
    layout := size(margobj, 2)

    glob ? x = xglo[which_point] : x = xloc[which_point]
    glob ? obj = fglo[which_point] : obj = floc[which_point]

    merge!(plotattributes, fonts())
    legend := :none
    # grid := false

    for k in axes(margobj, 2)
        @series begin
            subplot := k
            margobj[:, k, 1], margobj[:, k, 2]
        end
        @series begin
            subplot := k
            title := labs[k]
            seriestype := :scatter
            markercolor := "red"
            markersize := 3
            markershape := :circle
            markeralpha := 1.0
            markerstrokewidth := 1
            [x[k]], [obj]
        end
    end
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
- firstN: How many observations to plot.
"""
fsanity

@userplot FSanity

@recipe function f(h::FSanity; glob=h.args[2].npmm.onlyglo, firstN=1000, ylimss=fill(:none, 1 + length(h.args[2].pmm.momdat)))
    if length(h.args) != 2 || !(h.args[1] isa EstimationSetup) ||
       !(h.args[2] isa EstimationResult)
        error("fsanity should be given two inputs: an EstimationSetup and an EstimationResult. Got: $(typeof(h.args))")
    end
    estset, mmsolu = h.args
    @unpack mode, modelname = estset
    @unpack fglo, floc, xglo, xloc, conv, npmm = mmsolu

    glob ? ob = fglo : ob = floc
    glob ? x = xglo : x = xloc

    labs = labels(estset)
    lastindex = min(firstN, length(ob))
    xx = hcat([x[j] for j in 1:length(ob)]...)
    layout := size(xx, 1) + 1
    merge!(plotattributes, fonts())
    legendfontsize := 4

    @series begin
        subplot := 1
        label := ""
        title := "objective"
        titlefont := 10
        ylims := ylimss[1]
        ob[1:lastindex]
    end
    if !glob
        @series begin
            subplot := 1
            seriestype := :scatter
            label := "converged"
            color := :green
            markersize := 2.0
            collect(1:lastindex)[conv[1:lastindex]], ob[1:lastindex][conv[1:lastindex]]
        end
        @series begin
            subplot := 1
            seriestype := :scatter
            label := "not converged"
            color := :red
            markersize := 2.0
            collect(1:lastindex)[.!(conv[1:lastindex])], ob[1:lastindex][.!(conv[1:lastindex])]
        end
    end

    for k in axes(xx, 1)
        subplot := k + 1
        @series begin
            label := ""
            xx[k, 1:lastindex]
        end
        @series begin
            label := ""
            title := labs[k]
            color := :red
            linestyle := :dash
            ylims := ylimss[k+1]
            ones(lastindex) * xx[k, 1]
        end
    end
end

@userplot FGlobounds

@recipe function f(h::FGlobounds; N = 20)
    if length(h.args) != 2 || !(h.args[1] isa EstimationSetup) ||
       !(h.args[2] isa EstimationResult)
        error("fglobounds should be given two inputs: an EstimationSetup and an EstimationResult. Got: $(typeof(h.args))")
    end
    estset, mmsolu = h.args
    @unpack mode, modelname = estset
    @unpack xglo, npmm = mmsolu

    labs = labels(estset)

    xx = hcat(xglo...)

    m = length(npmm.sobolinds)^(1/N)
    percs = [1 / m^(i - 1) for i in 1:N]

    legend := :none
    merge!(plotattributes, fonts())
    layout := size(xx, 1)

    for k in axes(xx, 1)
        mins = [minimum(@view xx[k, 1:floor(Int, length(npmm.sobolinds) * percs[i])]) for i in 1:N]
        maxs = [maximum(@view xx[k, 1:floor(Int, length(npmm.sobolinds) * percs[i])]) for i in 1:N]
        @series begin
            subplot := k
            title := labs[k]
            color := :blue
            xaxis := :log10
            percs, mins
        end
        @series begin
            subplot := k
            color := :blue
            xaxis := :log10
            percs, maxs
        end
    end
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
function tableest_inner(estset::EstimationSetup, mmsolu::EstimationResult; glob::Bool=mmsolu.npmm.onlyglo)
    @unpack xglo, xloc, npmm = mmsolu

    labs = labels(estset)
    glob ? x = xglo[1] : x = xloc[1]

    return DataFrame(:Variable => labs, Symbol("Point estimate") => round.(x, digits=3))
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
function tableest(estset::EstimationSetup, mmsolu::EstimationResult; glob::Bool=mmsolu.npmm.onlyglo, saving::Bool=false, filename_suffix::String="")
    @unpack mode, modelname = estset
    @unpack npmm = mmsolu

    df = tableest_inner(estset, mmsolu; glob)

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
function tableest(estset::EstimationSetup, mmsolu::EstimationResult, boot::BootstrapResult; cilev::Real=0.05, dgt::Int64=3, glob::Bool=mmsolu.npmm.onlyglo, saving::Bool=false, filename_suffix::String="")
    @unpack mode, modelname = estset
    @unpack npmm = mmsolu
    @unpack xs, sd_asymp = boot

    df = tableest_inner(estset, mmsolu; glob)

    bs_sds = [sqrt(var(xs[i, :, :])) for i in axes(xs, 1)]
    ratios = [mean(var(xs[i, :, :]; dims=1)) / var(xs[i, :, :]) for i in axes(xs, 1)]
    bs_ci = [round.(quantile(xs[i, :, :][:], [cilev / 2, 1.0 - cilev / 2]), digits=dgt) for i in axes(xs, 1)]

    df[:, :("Asymptotic standard errors")] = round.(sd_asymp, digits=dgt)
    df[:, :("Bootstrapped standard errors")] = round.(bs_sds, digits=dgt)
    df[:, "Bootstrapped $(100*(1-cilev))% CI"] = bs_ci
    df[:, :("Seed share of variance")] = round.(ratios, digits=dgt)

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
function tablemoms_inner(estset::EstimationSetup, mmsolu::EstimationResult, which_point::Integer)
    @unpack mode, typemom = estset
    @unpack momloc, npmm = mmsolu

    momsdata = datamoments(mode, typemom) # moment from data to match

    df = momentnames(mode, typemom)

    df[:, :("Sample values")] = round.(momsdata[:, which_point], digits=3)
    df[:, :("Model values")] = round.(momloc[which_point], digits=3)

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

    df = tablemoms_inner(estset, mmsolu, 1)

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
function tablemoms(estset::EstimationSetup, mmsolu::EstimationResult, boot::BootstrapResult; dgt::Int64=3, saving::Bool=false, filename_suffix::String="")
    @unpack mode, modelname = estset
    @unpack npmm = mmsolu
    @unpack moms, W = boot

    df = tablemoms_inner(estset, mmsolu, 1)

    df[:, :("Simulated standard errors")] = round.(sqrt.(diag(Omega_boots(moms))), digits=dgt)
    df[:, :("Efficient weights (approx)")] = round.(diag(W), digits=dgt)

    saving && CSV.write(estimation_output_path() * estimation_name(estset, npmm, filename_suffix) * "_tablemoms_boot.csv", df)

    println(df)

    return df
end

"""
$(TYPEDSIGNATURES)

compares model-generated moments with data in a plot
"""
fmoms

@userplot FMoms

@recipe function f(h::FMoms; which_point = 1)
    if length(h.args) != 3 || !(h.args[1] isa EstimationSetup) ||
       !(h.args[2] isa EstimationResult) ||
       !(h.args[3] isa Integer)
        error("fmoms should be given three inputs: an EstimationSetup, an EstimationResult and an Integer. Got: $(typeof(h.args))")
    end
    estset, mmsolu, ff = h.args
    @unpack momloc, npmm = mmsolu

    df = tablemoms_inner(estset, mmsolu, which_point)

    if ncol(df) == 3
        insertcols!(df, 1, :empty => "Moments")
    end

    titles = unique(df[:, 1])
    xlb = names(df)[2]

    legend := :topleft
    size := (650, 540)
    foreground_color_legend := nothing
    xlabel := xlb
    xrotation := 45
    titlefont := 14
    titlelocation := :left
    xguidefont := 12
    yguidefont := 12
    tickfont := 10
    legendfont := 10

    # plot attributes
    lc = [:blue, :red]
    ls = [:solid, :dash]
    msh = [:circle, :x]
    msz = [5, 6]
    if ff == 1
        lb = ["Data", "Model"]
    else
        lb = ["", ""]
    end
    for k in 1:2
        @series begin

            label := lb[k]
            linecolor := lc[k]
            linestyle := ls[k]
            linewidth := 1.5
            markercolor := lc[k]
            markersize := msz[k]
            markershape := msh[k]
            markeralpha := 1.0
            markerstrokecolor := lc[k]
            markerstrokewidth := 2
            df[df[:, 1].==titles[ff], 2], df[df[:, 1].==titles[ff], 3:4][:, k]
        end
    end
end


function fmoms(estset::EstimationSetup, mmsolu::EstimationResult; which_point = 1, display_all = true)

    df = tablemoms_inner(estset, mmsolu, which_point)
    if ncol(df) == 3
        insertcols!(df, 1, :empty => "Moments")
    end
    titles = unique(df[:, 1])

    if display_all
        for i in eachindex(titles)
            display(fmoms(estset, mmsolu, i; which_point))
        end
    end

    return [fmoms(estset, mmsolu, i; which_point) for i in eachindex(titles)]
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
fbootstrap

@userplot FBootstrap

@recipe function f(h::FBootstrap; ci=true, cilev=0.05, trim=0.01)
    if length(h.args) != 3 || !(h.args[1] isa EstimationSetup) ||
       !(h.args[2] isa EstimationResult) ||
       !(h.args[3] isa BootstrapResult)
        error("fmoms should be given three inputs: an EstimationSetup, an EstimationResult and a BootstrapResult. Got: $(typeof(h.args))")
    end
    estset, mmsolu, boot = h.args
    @unpack npmm, xloc = mmsolu
    @unpack xs = boot

    labs = labels(estset)
    layout := size(xs, 1)
    merge!(plotattributes, fonts())

    for k in axes(xs, 1)
        xdist = sort(xs[k, :, :][:])
        toskip = floor(Int, length(xdist) * trim)
        xdist_trimmed = xdist[(1+toskip):(end-toskip)]

        subplot := k
        legend := :none

        @series begin
            seriestype := :histogram
            title := labs[k]
            xdist_trimmed
        end

        xopt = xloc[1][k]

        @series begin
            seriestype := :vline
            linecolor := "red"
            width := 2
            style := :dash
            [xopt]
        end
        if ci == true
            @series begin
                seriestype := :vline
                linecolor := "red"
                width := 1.5
                style := :dashdot
                quantile(xdist, [cilev / 2, 1.0 - cilev / 2], sorted=true)
            end
        end
    end
end

"""
$(TYPEDSIGNATURES)

Some plot attributes often used.
"""
function fonts()
    return Dict(
        :xrotation => 45,
        :titlefont => 8,
        :titlelocation => :left,
        :xguidefont => 6,
        :yguidefont => 6,
        :tickfont => 6,
        :legendfont => 6
    )
end