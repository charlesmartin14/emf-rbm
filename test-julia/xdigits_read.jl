using Boltzmann
using HDF5

function run_xdigits()

    X = h5open("xdigits.h5", "r") do file
        read(file, "X")
    end
    
    println("norm X ",norm(X))       
end

run_xdigits()

println("Press RETURN when ready")
readline(STDIN)

