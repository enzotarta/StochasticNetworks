# module LeNet
using MLDatasets
using Knet

const F = Float32
include("../../utility/common.jl")

function predict(w, x, bmom)
    i = 1
    x = reshape(x, 28, 28, 1, length(x)÷(28*28))
    
    x = conv4(w[i], x; padding=0)
    x = pool(x)
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    x = fSign.(x)
    i += 3
    
    x = conv4(w[i], x; padding=0)
    x = pool(x)
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    x = fSign.(x)
    i += 3
   
    x = mat(x)
   
    x = w[i]*x
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    x = fSign.(x)
    i += 3
    
    x = w[i]*x
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i += 3
    
    return x
end

function build_net(; atype=Array{F})
    w = [
      xavier(5,5,1,20),
      ones(1,1,20,1),
      zeros(1,1,20,1),
      
      xavier(5,5,20,50),
      ones(1,1,50,1),
      zeros(1,1,50,1),
      
      xavier(500,800),
      ones(500,1),
      zeros(500,1),
      
      xavier(10,500),
      ones(10,1),
      zeros(10,1),
    ]
    return map(a->convert(atype,a), w)
end

loss(w, x, y, bmom) = nll(predict(w, x, bmom), y)

binarize(w) = [i%3==1 ? sign.(w[i]) : w[i] for i=1:length(w)]

function main(xtrn, ytrn, xtst, ytst;
        seed = -1,
        batchsize = 100,
        lr = 1e-3,
        epochs = 100,
        infotime = 1,  # report every `infotime` epochs
        reportname = "",
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
        verb = 2
        )

    info("using ", atype)
    seed > 0 && srand(seed)

    w = build_net(atype=atype)
    opt = [Adam(lr=lr) for _=1:length(w)]
    bmom = [BatchMoments() for _=1:(length(w)÷3)]
    
    report(epoch) = begin
            bw = binarize(w)
            dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
            dtst = minibatch(xtst, ytst, batchsize; xtype=atype)
            acctrn = accuracy(bw, dtrn, (w,x)->predict(w,x,bmom))
            acctst = accuracy(bw, dtst, (w,x)->predict(w,x,bmom))
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
            bw = binarize(w)
            dw = grad(loss)(bw, x, y, bmom)
            update!(w, dw, opt)
            for i=1:3:length(w)
                w[i] = clip(w[i])
            end
        end
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
    end; toq()

    return w
end

# end # module
