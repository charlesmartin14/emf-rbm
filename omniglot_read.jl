using HDF5

function read_omniglot()

    Xtrain = h5open("omniglot.h5", "r") do file
        read(file, "train_x")
    end

    println("size Xtrain ",size(Xtrain))       
    println("norm Xtrain ",norm(Xtrain))

    Xtest = h5open("omniglot.h5", "r") do file
        read(file, "test_x")
    end

    println("size Xtest ",size(Xtest))       
    println("norm Xtest ",norm(Xtest))
    
end

read_omniglot()

