## USAGE

include("baseline/lenet_mnist.jl")
xytt = loadmnist(preprocess=true, fashion=false)
main(xytt..., seed=1, epochs=200, batchsize=200, lr=1e-3, pdrop=0.5)
