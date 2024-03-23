using Distributions
using LinearAlgebra
using StatsBase


"""
`Param` 

Mutable structure that encapsulates parameters for the gloabal evolutionary model.

# Fields
- `bf_mu::Float64`: Beneficial mutation rate within the host.
- `nt_mu::Float64`: Neutral mutation rate within the host.
- `bf_effect_dist::LogNormal{Float64}`: Distribution of beneficial mutation effects, modeled as a LogNormal distribution.
- `nt_effect_dist::Normal{Float64}`: Distribution of neutral mutation effects, modeled as a Normal distribution.
- `t_s::Float64`: Acute latent period in the epidemiological model.
- `mu_cd::Float64`: μ parameter of the LogNormal distribution for the chronic disease (cd) latent time.
- `sigmma_cd::Float64`: σ parameter of the LogNormal distribution for the chronic disease (cd) latent time.
- `cd_prob::Float64`: Probability of a chronic disease case among new offspring for each generation.
"""

mutable struct Param
    # Intra-host WF parameters
    bf_mu::Float64          # Beneficial mutation rate
    nt_mu::Float64          # Neutral mutation rate
    bf_effect_dist::LogNormal{Float64}  # Distribution of beneficial mutation effect
    nt_effect_dist::Normal{Float64}    # Distribution of neutral mutation effect
    # Epi model parameters
    t_s::Float64            # Acute latent period
    mu_cd::Float64          # μ parameter of the LogNormal dist for the cd latent time
    sigmma_cd::Float64      # σ parameter of the LogNormal dist for the cd latent time
    cd_prob::Float64        # Probability of a cd case among new offspring for each generation.
end

"""
`Param()` 

Constructor function for the `Param` structure, creating an instance with default values.

# Returns
- A `Param` instance with all fields initialized to zero.
"""
function Param()
    return Param(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
end

"""
`Species` 
Mutable structure representing a population within the host in an evolutionary model.

# Fields
- `n::Int64`: Number of members in the population.
- `s::Array{Float64, 1}`: Sequence of selection coefficients representing mutations.
- `f::Float64`: Fitness of the species.
- `a::Array{Float64, 1}`: Ancestor sequence of the species.
"""

mutable struct Species
    n::Int64                   
    s::Array{Float64, 1}       
    f::Float64                
    a::Array{Float64, 1}   
end

"""
`init_Species(number_of_members::Int64; mut_effect=[])` 

Initializes a `Species` instance within an evolutionary model.

# Arguments
- `number_of_members::Int64`: Number of members in the population.
- `mut_effect::Array{Float64, 1}`: Optional array representing the mutation effects. Default is an empty array.

# Returns
- A `Species` instance with the specified number of members and mutation effects.
"""

function init_Species(number_of_members::Int64; mut_effect=[])
    if isempty(mut_effect)
        return Species(number_of_members, [], 1.0, [])
    else
        return Species(number_of_members, mut_effect, 1.0 + sum(mut_effect), deepcopy(mut_effect))
    end
end


"""
`clone(s::Species)` 

Creates a new `Species` instance by cloning an existing one.

# Arguments
- `s::Species`: The species to be cloned.

# Returns
- A new `Species` instance with the same characteristics as the input species, but with `n` set to 1.
"""

function clone(s::Species)
    output = deepcopy(s)
    output.n = 1
    return output
end

"""
`mutate(s::Species, params::Param)` 

Mutates a given `Species` instance based on specified mutation rates and effects.

# Arguments
- `s::Species`: The species to be mutated.
- `params::Param`: A `Param` instance containing mutation-related parameters.

# Returns
- An array of new `Species` instances, representing the mutated population.
"""


function mutate(s::Species, params::Param)
    newSpecies = []

    if s.n > 0
        nMut_b = rand(Binomial(s.n, params.bf_mu))  # Number of individuals with beneficial mutation
        nMut_n = rand(Binomial(s.n, params.nt_mu))  # Number of individuals with neutral mutation

        s.n -= (nMut_b + nMut_n)  # Subtract number mutated from size of the current clone

        # Process mutations beneficial
        mut_effect_b = rand(params.bf_effect_dist, nMut_b)  # Choose mutation effect

        for i in 1:length(mut_effect_b)
            new_s = clone(s)  # Create a new copy of the sequence
            push!(new_s.s, mut_effect_b[i])   # Add mutation effect
            new_s.f = 1.0 + sum(new_s.s)  # Update fitness
            push!(newSpecies, new_s)
        end

        # Process mutations neutral
        mut_effect_n = rand(params.nt_effect_dist, nMut_n)  # Choose mutation effect

        for i in 1:length(mut_effect_n)
            new_s = clone(s)  # Create a new copy of the sequence
            push!(new_s.s, mut_effect_n[i])   # Add mutation effect
            new_s.f = 1.0 + sum(new_s.s)  # Update fitness
            push!(newSpecies, new_s)
        end
    end

    # Return the result
    if s.n > 0
        push!(newSpecies, s)
    end

    return newSpecies
end



"""
`evolve_pop_WF(T::Int64, N::Int64, params::Param, init_pop::Array{Species, 1})` 

Simulates the Wright-Fisher evolution of a population over a specified number of generations.

# Arguments
- `T::Int64`: The number of generations to simulate.
- `N::Int64`: The total population size.
- `params::Param`: A `Param` instance containing evolutionary and epidemiological parameters.
- `init_pop::Array{Species, 1}`: An array of initial `Species` instances representing the starting population.

# Returns
- An array of `Species` instances representing the evolved population after T generations.
"""


function evolve_pop_WF(T::Int64, N::Int64, params::Param, init_pop::Array{Species, 1})
    pop = deepcopy(init_pop)

    tStart = 1  # Start generation
    for t in tStart:T
        # Select species to replicate
        r = [s.n * s.f for s in pop]
        p = r / sum(r)  # Probability of selection for each species (sequence)
        n = rand(Multinomial(N, p))  # Selected number of each species

        # Update population size and mutate
        newPop = []
        for i in 1:length(pop)
            pop[i].n = n[i]  # Set new number of each species
            # Include mutations, then add mutants to the population array
            p = mutate(pop[i], params)
            for j in 1:length(p)
                push!(newPop, p[j])
            end
        end
        pop = newPop
    end

    return pop
end


# Between-host level functions

"""
`individual` is a mutable structure representing an individual in a between-host evolutionary and epidemiological model.

# Fields
- `id::Int64`: A unique identifier assigned to the individual based on the order of creation.
- `pop::Array{Species, 1}`: An array of `Species` instances representing the population of genetic variants associated with the individual.
- `t_init::Float64`: The initial time of contagion, indicating when the individual became infected.
- `t_c::Float64`: The incubation time, representing the duration between infection and becoming infectious.
- `infected::Bool`: A boolean flag indicating whether the individual is infected (true) or not (false).
- `variant::Array{Float64, 1}`: An array of mutations used to infect other individuals. Each element represents a mutation effect.
- `num_offspring::Int64`: The number of offspring produced by the individual during its infectious period.
- `ancestor::Int64`: The ID of the individual that served as the ancestor for the current individual.
"""

mutable struct individual
    id::Int64                    # ID number given by the order of creation
    pop::Array{Species, 1}       # Population of variants
    t_init::Float64              # Initial time of contagion
    t_c::Float64                 # Incubation time
    infected::Bool               # True or false
    variant::Array{Float64, 1}   # List of mutations used to infect other individuals
    num_offspring::Int64         # Number of offspring
    ancestor::Int64              # Given by the ancestor ID
end


"""
`init_individual(id::Int64, t0::Float64, t1::Float64, N::Int64, id_ancest::Int64; s_mutations=[])` 

Initializes an `individual` instance.

# Arguments
- `id::Int64`: A unique identifier assigned to the individual based on the order of creation.
- `t0::Float64`: The initial time of contagion, indicating when the individual became infected.
- `t1::Float64`: The incubation time, representing the duration between infection and becoming infectious.
- `N::Int64`: Total population size.
- `id_ancest::Int64`: The ID of the individual that served as the ancestor for the current individual.
- `s_mutations::Array{Float64, 1}`: Optional array representing the mutation effects. Default is an empty array.

# Returns
- An `individual` instance with the specified characteristics.
"""

function init_individual(id::Int64, t0::Float64, t1::Float64, N::Int64, id_ancest::Int64; s_mutations=[])
    return individual(id, [init_Species(N, mut_effect=s_mutations)], t0, t1, true, s_mutations, 0, id_ancest)
end


"""
`evolve_individual_pop(ind::individual, evol_time::Int64, N::Int64, params::Param)` 

Evolves the population associated with an `individual` using the Wright-Fisher model.

# Arguments
- `ind::individual`: The individual to be evolved.
- `evol_time::Int64`: The number of generations to simulate.
- `N::Int64`: Total population size.
- `params::Param`: A `Param` instance containing evolutionary and epidemiological parameters.

# Returns
- The updated `individual` instance with the evolved population.
"""

function evolve_individual_pop(ind::individual, evol_time::Int64, N::Int64, params::Param)
    ind.pop = evolve_pop_WF(evol_time, N, params, ind.pop)
    return ind
end


"""
`pick_rand_variant(ind::individual)` 

Randomly selects a variant from the population associated with an individual.

# Arguments
- `ind::individual`: The individual from which to pick a random variant.

# Returns
- A tuple `(variant, fitness)` representing the selected variant's genetic sequence (`variant`) and its associated fitness (`fitness`).
"""

function pick_rand_variant(ind::individual)
    pop_length = length(ind.pop)
    infector_virus_weights = zeros(pop_length)
    for i in 1:pop_length
        infector_virus_weights[i] = ind.pop[i].n
    end
    infector_virus_weights = infector_virus_weights / sum(infector_virus_weights)
    pos = wsample(1:length(infector_virus_weights), infector_virus_weights)

    return ind.pop[pos].s, ind.pop[pos].f
end


"""
function evolve_pop_EpiModel(case0::individual, tEnd::Int64, N::Int64, params::Param; chronical_cases=true)

Simulates the evolution of a population over time in an epidemiological model, considering chronic cases.

# Arguments
- `case0::individual`: The initial infected individual.
- `tEnd::Int64`: The end time of the simulation.
- `N::Int64`: Intrah host  population size.
- `params::Param`: Parameters defining the evolutionary and epidemiological model.
- `chronical_cases::Bool=true`: Flag indicating whether chronic cases are considered (default is true).

# Returns
- `pop_final::Vector{Vector{individual}}`: Population history over time, represented as an array of arrays of individuals.
- `num_sick::Vector{Int64}`: Number of infected individuals at each time step.
- `rare_events::Vector{Int64}`: Number of rare events (e.g., chronic cases) at each time step.
- `chronical_times::Vector{Float64}`: Times of occurrence for chronic cases.

# Description
The function simulates the spread of an infection in a population, considering the possibility of chronic cases.
It evolves the population over the specified time period, with each individual's infection status, mutations, and offspring generation updated at each time step.
"""
function evolve_pop_EpiModel(case0::individual, tEnd::Int64, N::Int64, params::Param; chronical_cases=true)
    # CD time distribution
    μ = params.mu_cd  # Mean of the associated Normal distribution
    σ = params.sigmma_cd  # Std of the associated Normal distribution
    d = LogNormal(μ, σ)

    # Initial super-spreading parameters
    k = 1.0
    R = 100.0

    tStart = 1  # Start generation
    id = 2

    indiv_pop = []
    push!(indiv_pop,deepcopy(case0))
    pop_final=[]
    push!(pop_final,deepcopy(indiv_pop))  
    

    num_sick = zeros(tEnd + 1)
    num_sick[1] = 1
    rare_events = zeros(tEnd + 1)
    chronical_times = []
    rare_events[1] = 0

    for t in tStart:tEnd
        num_sick[t + 1] = length(indiv_pop)
        push!(pop_final, deepcopy(indiv_pop))
        total_population = length(indiv_pop)
        tot_num_offspring = 0.0
        offspring_pop = []

        for i in 1:length(indiv_pop)
            delta_t = Int64(t - indiv_pop[i].t_init)

            if delta_t < indiv_pop[i].t_c
                rep_per_day = 2  # Number of replications per day
                evolve_individual_pop(indiv_pop[i], rep_per_day, N, params)
                push!(offspring_pop, indiv_pop[i])
            else
                av_fitness = 0.0

                for indiv in indiv_pop[i].pop
                    av_fitness += indiv.f
                end

                indiv_pop[i].infected = false
                av_fitness = av_fitness / length(indiv_pop[i].pop)
                R_a = R * (1 + av_fitness) / length(indiv_pop)

                p = k / (k + R_a)
                num_offspring = rand(NegativeBinomial(k, p))  # Super-spreading
                indiv_pop[i].num_offspring = num_offspring
                tot_num_offspring += num_offspring

                for j in 1:num_offspring
                    if chronical_cases
                        if rand() < params.cd_prob
                            rare_events[t + 1] += 1
                            T_WF = rand(d)  # Chronic time
                            push!(chronical_times, T_WF)
                        else
                            T_WF = params.t_s  # Standard time
                        end
                    else  
                        T_WF = params.t_s  # Standard time
                    end  


                    (seq, fitness) = pick_rand_variant(indiv_pop[i])
                    push!(offspring_pop, init_individual(id, Float64(t), T_WF, N, indiv_pop[i].id, s_mutations=seq))
                    id += 1
                end
            end
        end

        indiv_pop = deepcopy(offspring_pop)
    end

    return pop_final, num_sick, rare_events, chronical_times
end

