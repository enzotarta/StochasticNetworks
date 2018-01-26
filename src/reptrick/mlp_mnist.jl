using MLDatasets
using Knet

const F = Float32
include("../../utility/common.jl")

function predict(w, x, bmom; pdrop=0.5)
    i = 1
    x = mat(x)
    
    x = w[i]*x
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    x = fSign.(x)
    x = dropout(x, pdrop)
    i += 3
    
    x = w[i]*x
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    x = fSign.(x)
    x = dropout(x, pdrop)
    i += 3
    
    x = w[i]*x
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    x = fSign.(x)
    x = dropout(x, pdrop)
    i += 3
    
    x = w[i]*x
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i += 3

    return x
end

loss(w, x, y, bmom; pdrop=0.5) = nll(predict(w, x, bmom; pdrop=0.5), y) 

binarize(w) = [i%3==1 ? sign.(w[i]) : w[i] for i=1:length(w)]

function build_net(; atype=Array{F})
    w = [
      xavier(800, 28*28),
      ones(800, 1),
      zeros(800, 1),

      xavier(800, 800),
	  ones(800, 1),
      zeros(800, 1),

      xavier(800, 800),
	  ones(800, 1),
      zeros(800, 1),

      xavier(10, 800),
	  ones(10, 1),
      zeros(10, 1),
    ]
    return map(a->convert(atype,a), w)
end

loss(w, x, y, bmom; pdrop=0.5) = nll(predict(w, x, bmom; pdrop=pdrop), y)

function main(xtrn, ytrn, xtst, ytst;
    seed = -1,
    batchsize = 200,
    lr = 1e-3,
    epochs = 200,
    infotime = 1,  # report every `infotime` epochs
    reportname = "",
    atype = gpu() >= 0 ? KnetArray{F} : Array{F},
    verb = 2,
    pdrop = 0.5
    )

info("using ", atype)
seed > 0 && srand(seed)

w = build_net(atype=atype)
opt = [Adam(lr=lr) for _=1:length(w)]
bmom = [BatchMoments() for _=1:(length(w)÷3)]

acctrn = 0
acctst = 0

report(epoch) = begin
        dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
        dtst = minibatch(xtst, ytst, batchsize; xtype=atype)
        bw = binarize(w) 
        acctrn = accuracy(bw, dtrn, (w,x)->predict(w,x, bmom))
        acctst = accuracy(bw, dtst, (w,x)->predict(w,x, bmom)) 
        println((:epoch, epoch,
            :trn, accuracy(w, dtrn, (w,x)->predict(w,x, bmom)) |> percentage,
            :trn_clip, acctrn  |> percentage,
            :tst, accuracy(w, dtst, (w,x)->predict(w,x,bmom))  |> percentage,
            :tst_clip, acctst  |> percentage
        ))

        if reportname != ""
            open(reportname, "a") do f 
                print(f, epoch, "\t", acctrn, "\t", acctst, "\t")
                for lay = 1:3:length(w)
                    print(f, vecnorm(w[lay])/√length(w[lay]), "\t", mean(1 - w[lay].*w[lay]), "\t")
                end
                println(f)
            end
        end

        if verb > 1
            for i=1:3:length(w)
                n, m = length(w[i]), length(w[i+1])
                print(" layer $(i÷3+1): W-norm $(round(vecnorm(w[i])/√n, 3))")
                print(" Θ1-norm $(round(vecnorm(w[i+1])/√m, 3))")
                print(" Θ2-norm $(round(vecnorm(w[i+2])/√m, 3))\n")
            end
        end    
    end

    report(0); tic()
    @time for epoch=1:epochs
        for (x, y) in  minibatch(xtrn, ytrn, batchsize, shuffle=true, xtype=atype)
            dw = grad(loss)(w, x, y, bmom; pdrop=pdrop)
            update!(w, dw, opt)
            for i=1:3:length(w)
                w[i] = clip(w[i], 1e-2)
            end
        end
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
        # acctrn == 100 && break
    end; toq()
    println("# FINAL RESULT acccuracy: train=$acctrn test=$acctst" , )
    return w
end
