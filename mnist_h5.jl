using MNIST
using Boltzmann
using HDF5

const PATH = "./mnist.h5"

function save_data(filename::AbstractString, X::Array{Float64,2}, y::Array{Float64,1})
    h5open(filename,"w") do file
        write(file, "$(name)___X", X)
        write(file, "$(name)___y", y)
    end
end

# Load MNIST Training data
X, y = traindata()              # Raw data and labels
normalize_samples!(X)           # Pin to range [0,1]
binarize!(X;threshold=0.001)    # Create binary data

if isfile(PATH)
  rm(PATH)
end
save_data(PATH, X, y)
    
println(size(X))
println(norm(X))

println("Press RETURN when ready")
readline(STDIN)

