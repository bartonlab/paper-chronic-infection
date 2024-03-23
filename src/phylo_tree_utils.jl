using Phylo


evolve(tree) = Phylo.map_depthfirst((val, node) -> val + randn(), 0., tree, Float64)



"""
`mutable struct OTnode`:
  Represents a node in an ordered tree (OT).

  Fields:
  - `par::Int64`: Parent node ID. Set to -1 if the node is the root.
  - `children::Array{Int64,1}`: Array of children node IDs.
  - `dt::Float64`: Branch respect the parent node.
  - `conf::Array{Int64,1}`: node configuration.
"""

mutable struct OTnode
	par::Int64
	children::Array{Int64,1}
	dt::Float64
	conf::Array{Int64,1}
end

"""
`function ReadTree(trin::Array{Float64,2})`:
  Reads a tree file `trin` representing an ordered tree.

  Parameters:
  - `trin::Array{Float64,2}`: array where each row corresponds to a node in the ordered tree. Columns represent:
    1. Node ID
    2. Parent node ID
    3. Branch length or time duration

  Returns:
  - `tree::Array{OTnode,1}`: Array of OTnode structures representing the ordered tree.

  Description:
  - Initializes an array `tree` to store OTnode structures.
  - Iterates through each row of `trin` to create and configure OTnodes.
  - Assigns parent-child relationships and branch lengths.
  - Returns the array representing the ordered tree.
"""

function ReadTree(trin::Array{Float64,2})
  N = size(trin)[1]
  tree = Array{OTnode,1}(undef,N)
  for (i,n) in enumerate(trin[:,1])
    n = Int64(n)
    tree[n] = OTnode(-1, Array{Int64,1}(undef,0), -1., Array{Int64,1}(undef,0))
    tree[n].par = Int64(trin[i,2])
    if tree[n].par >= 0
      push!(tree[tree[n].par].children,n)
    end
    tree[n].dt = trin[i,3]
  end
  return tree
end
"""
`function get_newick_subtree(tree::Vector{OTnode}, node_id::Int64)`:
  Recursively generates a Newick subtree representation starting from a given node in the tree.

  Parameters:
  - `tree::Vector{OTnode}`: Vector of OTnode structures representing the ordered tree.
  - `node_id::Int64`: ID of the current node for Newick subtree generation.

  Returns:
  - Newick subtree representation as a string.

  Description:
  - Retrieves the node information from the tree based on `node_id`.
  - If the node has no children, returns a string in the format "node_id:branch_length".
  - If the node has children, recursively generates Newick representations for each child.
  - Joins the child representations with commas and constructs the final subtree string.
"""

function get_newick_subtree(tree::Vector{OTnode}, node_id::Int64)
    node = tree[node_id]
    if length(node.children) == 0
        return string(node_id, ":", node.dt)
    else
        children_str = join([get_newick_subtree(tree, child) for child in node.children], ",")
        return "($children_str):$(node.dt)"
    end
end

"""
`function tree_to_newick(tree::Vector{OTnode})`:
  Generates the Newick representation of the entire ordered tree.

  Parameters:
  - `tree::Vector{OTnode}`: Vector of OTnode structures representing the ordered tree.

  Returns:
  - Newick representation as a string.

  Description:
  - Finds the root node ID by searching for the node with `par == -1`.
  - Raises an error if no root is found.
  - Calls `get_newick_subtree` with the root node ID to obtain the Newick representation of the entire tree.
  - Appends a semicolon to the result and returns the final Newick string.
"""


function tree_to_newick(tree::Vector{OTnode})
    root_id = findfirst(node -> node.par == -1, tree)
    if root_id === nothing
        error("Tree does not have a root.")
    end
    return get_newick_subtree(tree, root_id) * ";"
end




