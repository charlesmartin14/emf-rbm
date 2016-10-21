using Boltzmann
using MNIST

function run_mnist()
    # Set parameters
    Epochs         = 20
    HiddenUnits    = 256
    Approx         = "tap2"
    ApproxSteps    = 3
    LearnRate      = 0.005
    MonitorEvery   = 5
    PersistStart   = 5
    Momentum       = 0.5
    DecayMagnitude = 0.01
    DecayType      = "l1"

    # Load MNIST Training data
    X, y = traindata()              # Raw data and labels
    normalize_samples!(X)           # Pin to range [0,1]
    binarize!(X;threshold=0.001)    # Create binary data
    # Hold out a validation set
    TrainSet     = X[:,1:50000]
    ValidSet     = X[:,50001:end]

    # Initialize Model
    m = BernoulliRBM(28*28, HiddenUnits,(28,28); 
                     momentum  = Momentum, 
                     TrainData = TrainSet,
                     sigma     = 0.001)

    # Run Training
    rbm,monitor = fit(m, TrainSet; n_iter           = Epochs, 
                                   weight_decay     = DecayType,
                                   decay_magnitude  = DecayMagnitude,
                                   lr               = LearnRate,
                                   persistent       = false,
                                   validation       = ValidSet,
                                   NormalizationApproxIter = ApproxSteps,
                                   monitor_every    = MonitorEvery,
                                   monitor_vis      = true,
                                   approx           = Approx,
                                   persistent_start = PersistStart)

    WriteMonitorChartPDF(rbm,monitor,X,"mnist.tap2.fp.a3.pdf")
    save_params("mnist.tap2.fp.a3.h5",rbm,"mnist");
end

run_mnist()

println("Press RETURN when ready")
readline(STDIN)

