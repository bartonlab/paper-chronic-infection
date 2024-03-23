

using SavitzkyGolay
using ChangePointDetection

"""
function av_number_of_mut(Repports::Array{Any, 1})

Compute the average number of mutations in populations at each time.

# Arguments
- `Repports::Array{Any, 1}`: Array containing reports of populations over time.

# Returns
- `fraction_mut::Vector{Float64}`: Average number of mutations at each time point.

# Description
This function calculates the average number of mutations in populations at each time point. It iterates over the provided reports, considering each individual in the population. For each individual, it calculates the weighted average of mutations, where the weight is the abundance of the species. The final result is the average number of mutations across all individuals at each time point.
"""


function av_number_of_mut(Repports::Array{Any, 1})
    time = length(Repports)
    fraction_mut = zeros(time)
    for t in 1:time
        indiv_t = Repports[t]
        for i in 1:length(indiv_t)
            pop = indiv_t[i].pop
            n_spec = map(x -> x.n, pop)
            n_mut = map(x -> length(x.s), pop)
            fraction_mut[t] += sum(n_spec .* n_mut) / sum(n_spec)
        end
        fraction_mut[t] = fraction_mut[t] / length(indiv_t)
    end
   
    return fraction_mut
end


"""
function av_number_of_mut_variant(Repports::Array{Any, 1})

Compute the average number of mutations in transmitted variants at each time.

# Arguments
- `Repports::Array{Any, 1}`: Array containing reports of populations over time.

# Returns
- `fraction_mut::Vector{Float64}`: Average number of mutations in transmitted variants at each time point.

# Description
This function calculates the average number of mutations in transmitted variants at each time point. It iterates over the provided reports and computes the mean number of mutations for the transmitted variants in each report.
"""
function av_number_of_mut_variant(Repports::Array{Any, 1})
    time = length(Repports)
    fraction_mut = zeros(time)
    
    for t in 1:time
        fraction_mut[t] = mean(length.(map(x -> x.variant, Repports[t])))
    end
    
    return fraction_mut
end

 """

 function slope_change_score(y::Array{Float64,1})

Compute the slope at each time and its maximum.

# Arguments
- `y::Array{Float64, 1}`: Array of values representing a curve over time.

# Returns
- `m::Array{Float64, 1}`: Array of slopes at each time.
- `max_slope::Float64`: Maximum slope of the curve.

# Description
This function calculates the slope at each time point and its maximum value for a given curve represented by the array `y`. It uses finite differences to compute the slope and returns both the array of slopes (`m`) and the maximum slope (`max_slope`).
"""  

function slope_change_score(y::Array{Float64,1})
    x=1:1:length(y);
    m = diff(y, dims=1)./diff(x, dims=1);
    return m,maximum(m)
end    


"""
function gaussian_weight(data_range::Vector{Int64},t0::Int64,sigma::Int64)

Compute the Gaussian weight normalized for the interval 1:data_range.

# Arguments
- `data_range::Vector{Int64}`: Vector representing the range of data.
- `t0::Int64`: Central position of the Gaussian distribution.
- `sigma::Int64`: Standard deviation of the Gaussian distribution.

# Returns
- `output::Vector{Float64}`: Vector of Gaussian weights normalized for the given interval.

# Description
This function calculates the Gaussian weights for the specified `data_range`, centered at `t0` with standard deviation `sigma`. The resulting weights are normalized to ensure their sum equals 1.
"""

function gaussian_weight(data_range::Vector{Int64},t0::Int64,sigma::Int64)
    output=map(t-> exp(-(t-t0)^2/(2*sigma^2)),data_range)
    return output/sum(output)
end    



"""
function gaussian_smothing(x::Vector{Float64},sigma::Int64)

Smooth the signal "x" via Gaussian smoothing with a time window "sigma".

# Arguments
- `x::Vector{Float64}`: Signal to be smoothed.
- `sigma::Int64`: Time window for Gaussian smoothing.

# Returns
- `y::Vector{Float64}`: Smoothed signal.

# Description
This function applies Gaussian smoothing to the input signal "x" using a specified time window "sigma". It returns the smoothed signal "y".
"""


function gaussian_smothing(x::Vector{Float64},sigma::Int64)
    data_range=1:1:length(x)
    y=zeros(length(x))
    
    for t0 in data_range
        gw=gaussian_weight(Int64.(data_range),t0,sigma)
        y[t0]=gw'*x
    end
    
    return y
end    


"""
function detect_jump(fraction_mut_variant::Vector{Float64},μ_null::Float64,σ_null::Float64,sm_time_w::Int64,sm_poli_order::Int64)

Detect change points in a signal using a specified method.

# Arguments
- `fraction_mut_variant::Vector{Float64}`: Signal for change point detection.
- `μ_null::Float64`: Mean of the null distribution for z-scoring.
- `σ_null::Float64`: Standard deviation of the null distribution for z-scoring.
- `sm_time_w::Int64`: Time window for smoothing the signal.
- `sm_poli_order::Int64`: Polynomial order for Savitzky–Golay smoothing.

# Returns
- `jump_times::Vector{Float64}`: Detected change point times.

# Description
This function detects change points in the input signal "fraction_mut_variant" using a specified method. It involves smoothing the signal, calculating z-scores, and identifying positions with z-scores exceeding a threshold. Change point times are then determined from these positions.
"""

function detect_jump(
    fraction_mut_variant::Vector{Float64},
    μ_null::Float64,
    σ_null::Float64,
    sm_time_w::Int64,
    sm_poli_order::Int64
)
    # Outliers threshold
    th = 3.5

    # Compute the slope of the fraction_mut_variant vs t curve
    (slope_vs_t, score) = slope_change_score(fraction_mut_variant)

    # Smooth this slope with Savitzky–Golay filter
    sg = savitzky_golay(append!(slope_vs_t, 0.0), sm_time_w, sm_poli_order)
    sg = savitzky_golay(sg.y, sm_time_w, sm_poli_order)

    # Compute z-score for smoothed slope values from null distribution parameters "μ_null" and "μ_null"
    zs_ = (sg.y .- μ_null) ./ σ_null

    # Select times where z-score is higher than threshold "th"
    pos_with_th_score_ = zs_ .> th

    # Detect change point times from "pos_with_th_score_"
    # This will produce the starting time and end time for each jump
    profile = lsdd_profile(Float64.(pos_with_th_score_); window = 1)
    points = ChangePointDetection.getpoints(profile)

    # Compute the number of jumps and the time with max. value for each peak
    if length(points) % 2 == 0
        number_of_jumps = Int64(length(points) / 2)
    else
        number_of_jumps = Int64(floor(length(points) / 2)) + 1
    end

    # List of times for each jump in "jump_times", so the number of jumps will be its length
    jump_times = zeros(number_of_jumps)

    for i in 1:1:number_of_jumps
        if length(points) % 2 != 0 && i == number_of_jumps
            jump_times[i] = 0.5 * (last(points) + length(slope_vs_t))
        else
            jump_times[i] = mean(points[(i - 1) * 2 + 1:i * 2])
        end
    end

    return jump_times
end


"""
function fract_of_burst_per_cd(burst_times::Vector{Float64}, num_cd_pat_vs_t::Vector{Float64})

Compute the fraction of bursty transmissions per chronic case.

# Arguments
- `burst_times::Vector{Float64}`: Times of bursty transmissions.
- `num_cd_pat_vs_t::Vector{Float64}`: Number of chronic cases at each time.

# Returns
- `fraction::Float64`: Fraction of bursty transmissions per chronic case.

# Description
This function calculates the fraction of bursty transmissions per chronic case. If the input "burst_times" is empty, the fraction is 0.0. Otherwise, it computes the fraction by dividing the length of "burst_times" by the cumulative sum of chronic cases up to the last time in "burst_times".
"""
function fract_of_burst_per_cd(burst_times::Vector{Float64}, num_cd_pat_vs_t::Vector{Float64})
    if isempty(burst_times)
        return 0.0
    end

    las_t = Int64.(round(last(burst_times)))
    causal_cd = sum(num_cd_pat_vs_t[1:las_t])

    if causal_cd == 0.0
        return 0.0
    else
        return length(burst_times) / causal_cd
    end
end   



########## To produce Muller plot input files


"""
Create a dictionary mapping variants to identity numbers.

# Arguments
- `all_variants::Vector{Any}`: Vector containing variants.

# Returns
- `sqs_D::Dict{Vector{Float64}, Int64}`: Dictionary mapping variants to identity numbers.

# Description
This function creates a dictionary, `sqs_D`, that maps each variant in the input vector `all_variants` to a unique identity number.
"""
function dict_mut_list(all_variants::Vector{Any})
    sqs_D = Dict{Vector{Float64}, Int64}()
    for i in 1:length(all_variants)
        push!(sqs_D, all_variants[i] => i)
    end

    return sqs_D
end



"""
function muller_data(Reports::Vector{Any})

Create data frames for Muller plots by tracking each individual variant.

# Arguments
- `Reports::Vector{Any}`: Vector of populations over time.
# Returns
- `df::DataFrame`: Data frame containing information about the population, with columns Generation, Identity, and Population.
- `df_adj::DataFrame`: Adjacency data frame with columns Parent and Identity.
- `df_variant_sz::DataFrame`: Data frame containing information about the length of each identity, with columns Identity and Variant_size.

# Description
This function processes the input reports to create data frames suitable for generating Muller plots. The data frames include information about the population, adjacency, and variant sizes.
"""
function muller_data(Reports::Vector{Any})
    T = length(Reports)

    # Create a dictionary mapping unique variants to identity numbers
    all_variants_all_times = []
    for t in 1:T
        indiv_pop_at_t = Reports[t]
        for i in 1:length(indiv_pop_at_t)
            species_pop = indiv_pop_at_t[i].pop
            for sp in species_pop
                push!(all_variants_all_times, sp.s)
            end
        end
    end

    all_variants_all_times = unique(all_variants_all_times)
    sqs_to_identity_Dict = dict_mut_list(all_variants_all_times)
    sqs = sort(collect(sqs_to_identity_Dict), by=x -> x[2])

    # Create a dictionary mapping unique variants to species
    species_dict_list = [Dict(sp.s => sp) for t in 1:T for i in Reports[t] for sp in i.pop]
    unique_species_dict_list = Dict{Any, Any}()
    seen_keys = Set{Any}()

    for species_dict in species_dict_list
        for (key, value) in species_dict
            if key in seen_keys
                # Skip this element as it has a duplicate key
                continue
            end

            push!(seen_keys, key)
            unique_species_dict_list[key] = value
        end
    end

    # Create an adjacency data frame with Identity and Parents fields
    num_unique_seq = length(sqs_to_identity_Dict) - 1
    df_adj = DataFrame(
        Parent = zeros(Int64, num_unique_seq),
        Identity = zeros(Int64, num_unique_seq)
    )
    row = 1
    for sq in sqs[2:end]
        df_adj[row, :Identity] = sqs_to_identity_Dict[first(sq)]
        df_adj[row, :Parent] = sqs_to_identity_Dict[unique_species_dict_list[first(sq)].a]
        row += 1
    end

    # Create a data frame for the length of each identity
    df_variant_sz = DataFrame(
        Identity = zeros(Int64, length(sqs_to_identity_Dict)),
        Variant_size = zeros(Float64, length(sqs_to_identity_Dict))
    )
    row = 1
    for sq in sqs[1:end]
        df_variant_sz[row, :Identity] = sqs_to_identity_Dict[first(sq)]
        df_variant_sz[row, :Variant_size] = length(first(sq))
        row += 1
    end

    # Create the population data frame with Generation, Identity, and Population fields
    number_of_variants = length(all_variants_all_times)
    num_rows = T * number_of_variants
    df = DataFrame(
        Generation = zeros(Int64, num_rows),
        Identity = zeros(Int64, num_rows),
        Population = zeros(Float64, num_rows)
    )

    for t in 1:T
        indiv_pop_at_t = Reports[t]
        species_dict_list = [Dict(sp.s => sp) for sp in vcat((i.pop for i in indiv_pop_at_t)...)]
        for id in 1:length(sqs)
            df[(t - 1) * number_of_variants + id, :Generation] = t
            df[(t - 1) * number_of_variants + id, :Identity] = sqs[id].second
        end

        for (i, my_dict) in enumerate(species_dict_list)
            for val in values(my_dict)
                variant = val.s
                abundance = val.n
                id = sqs_to_identity_Dict[variant]
                df[(t - 1) * number_of_variants + id, :Population] += abundance
            end
        end
    end

    return df, df_adj, df_variant_sz
end
   



"""
function muller_data_mutations(Reports::Vector{Any})

Create data frames for Muller plots by tracking each consensus sequence number of mutations.

# Arguments
- `Reports::Vector{Any}`: Vector of reports containing information about the population dynamics.

# Returns
- `df::DataFrame`: Data frame containing information about the population, with columns Generation, Identity, and Population.
- `df_adj::DataFrame`: Adjacency data frame with columns Parent and Identity.

# Description
This function processes the input reports to create data frames suitable for generating Muller plots based on the consensus sequence number of mutations. The data frames include information about the population and adjacency.
"""
function muller_data_mutations(Reports::Vector{Any})
    length_variant_per_t = []
    identity_list = []
    for t in 1:length(Reports)
        indiv_pop_at_t = Reports[t]
        pop_size = length(indiv_pop_at_t)
        length_variant = zeros(pop_size)
        for i in 1:pop_size
            r = [s.n * s.f for s in indiv_pop_at_t[i].pop]
            (val, pos) = findmax(r)
            length_variant[i] = length(indiv_pop_at_t[i].pop[pos].s)
        end

        B = [(i, count(==(i), length_variant) / pop_size) for i in unique(length_variant)]
        push!(identity_list, unique(length_variant))
        push!(length_variant_per_t, B)
    end

    ident = unique(vcat(identity_list...))
    ful_md = []
    for t in 1:length(length_variant_per_t)
        ful_md_at_t = []
        pop_at_t = length_variant_per_t[t]
        identity_at_t = []
        for i in 1:length(pop_at_t)
            push!(ful_md_at_t, pop_at_t[i])
            push!(identity_at_t, pop_at_t[i][1])
        end

        for j in ident
            if isempty(findall(x -> x == j, vcat(identity_at_t...)))
                push!(ful_md_at_t, (j, 0.0))
            end
        end

        push!(ful_md, ful_md_at_t)
    end

    generation = []
    identity = []
    population = []
    for t in 1:length(ful_md)
        pop_at_t = ful_md[t]
        for j in 1:length(pop_at_t)
            push!(generation, t)
            push!(identity, pop_at_t[j][1])
            push!(population, pop_at_t[j][2])
        end
    end

    childs = []
    parents = []
    for i in 2:length(ident)
        push!(childs, ident[i])
        push!(parents, ident[i - 1])
    end

    df_adj = DataFrame(
        Parent = Int.(parents),
        Identity = Int.(childs),
    )

    df = DataFrame(
        Generation = Int.(generation),
        Identity = Int.(identity),
        Population = Float64.(population)
    )

    return df, df_adj
end
    













