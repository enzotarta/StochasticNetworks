using MLDatasets
using Knet
const F = Float32

include("../../utility/common.jl")

function predict(w, x, bmom; pdrop=0.5, input_do = 0.0)
    x = dropout(x, input_do; training= w[1] isa Rec ? true : false)

    x = mat(x)
    
		for i=1:2:length(w)-2
    	x = w[i]*x .+ w[i+1]
    	x = relu.(x)
    	x = dropout(x, pdrop)
    end
    return w[end-1]*x .+ w[end]
end

function build_net(; atype=Array{F})
    w = [
        xavier(800, 784),
        zeros(800,1),
        
        xavier(800, 800),
        zeros(800,1),
        
        xavier(800,800),
        zeros(800,1),
        
        xavier(10,800),
        zeros(10,1),
      ]
    return map(a->convert(atype,a), w)
end

loss(w, x, y, bmom; pdrop=0.5, input_do = 0.0) = nll(predict(w, x, bmom; pdrop=pdrop, input_do = input_do), y)

function main(xtrn, ytrn, xtst, ytst;
        seed = -1,
        batchsize = 200,
        lr = 1e-3,
        epochs = 200,
        infotime = 1,  # report every `infotime` epochs
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
				reportname = "",
        pdrop = 0.0,  #dropout probability
        input_do = 0.0
        )

    info("using ", atype)
    seed > 0 && srand(seed)

    w = build_net(atype=atype)
    opt = [Adam(lr=lr) for _=1:length(w)]
    bmom = [BatchMoments() for _=1:length(w)÷3]

    acctrn = 0
    acctst = 0

    dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
    dtst = minibatch(xtst, ytst, batchsize; xtype=atype)

    report(epoch) = begin
            acctrn = accuracy(w, dtrn, (w,x) ->predict(w,x,bmom))
            acctst = accuracy(w, dtst, (w,x) ->predict(w,x,bmom)) 
            println((:epoch, epoch,
                :trn, acctrn |> percentage,
                :tst, acctst |> percentage
                ))
      
            if reportname != ""
                open(reportname, "a") do f 
                    print(f, epoch, "\t", acctrn, "\t", acctst, "\t")
                    for lay = 1:3:length(w)
                        print(f, vecnorm(w[lay])/√length(w[lay]), "\t")
                    end
                    println(f)
                end
            end
        end

    report(0)
    for epoch=1:epochs
				ξ = minibatch(xtrn, ytrn, batchsize, shuffle=true, xtype=atype)
        for (x, y) in ξ
            dw = grad(loss)(w, x, y, bmom; pdrop=pdrop, input_do = input_do)
            update!(w, dw, opt)
        end
        (epoch % infotime == 0) && report(epoch)
    end
    println("# FINAL RESULT acccuracy: train=$acctrn test=$acctst" , )
    return w, acctst
end
