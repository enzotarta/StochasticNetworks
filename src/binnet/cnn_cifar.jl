using Knet
using MLDatasets

const F = Float32
include("../../utility/common.jl")

# all cnn are "same" 3x3
NET = [128,128, 'M', 256, 256, 'M',
  512, 512,  'M', 'F', [1024, 1024]]


function predict(w, x, bmom; pdrop=0.5)
    i = 1
    x = reshape(x, 32, 32, 3, length(x)÷(32*32*3))

    for idx = 1:length(NET)
        NET[idx] == 'F' && break
        if NET[idx] != 'M'
            x = conv4(w[i], x; padding=1)
            if NET[idx+1] == 'M'
                x = pool(x; mode=1)
            end
            x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
            x = fSign.(x)
            i += 3
        end
    end
    x = mat(x)
    for f in NET[end]
        x = w[i] * x
        x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
        x = fSign.(x)
        x = dropout(x, pdrop)
        i += 3
    end

    x = w[i] * x
    x = batchnorm(w[i+1:i+2], x, bmom[i÷3+1])
    i+= 3
    return x
end

loss(w, x, y, bmom; pdrop=0.0, input_do = 0.0) = nll(predict(w, dropout(x, input_do; training=true), bmom; pdrop=pdrop), y)

function build_net(; atype=Array{F})
    w = []
    f0 = 3
    for f in NET
        f == 'M' && continue
        f == 'F' && break
        push!(w, xavier(3,3,f0, f))
        push!(w, ones(1,1,f,1))
        push!(w, zeros(1,1,f,1))
        f0 = f
    end
    side = 32 ÷ 2^count(x->x=='M', NET)
    f0 = f0 * side^2
    for f in NET[end]
        push!(w, xavier(f,f0))
        push!(w, ones(f,1))
        push!(w, zeros(f,1))
        f0 = f
    end
    push!(w, xavier(10,f0))
    push!(w, ones(10,1))
    push!(w, zeros(10,1))
    return map(a->convert(atype,a), w)
end

binarize(w) = [i%3==1 ? sign.(w[i]) : w[i] for i=1:length(w)]

function main(xtrn, ytrn, xtst, ytst;
        seed = -1,
        batchsize = 200,
        lr = 1e-3,
        epochs = 200,
        infotime = 1,  # report every `infotime` epochs
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
				reportname = "",
        pdrop = 0.5,  #dropout probability
				input_do = 0.0,
        verb = 2
        )

    info("using ", atype)
    seed > 0 && srand(seed)

xtrn = 2xtrn - 1
xtst = 2xtst - 1

    w = build_net(atype=atype)
    opt = [Adam(lr=lr) for _=1:length(w)]
    bmom = [BatchMoments() for _=1:length(w)÷3]

    acctrn = 0
    acctst = 0
    dtrn = minibatch(xtrn, ytrn, batchsize; xtype=atype)
    dtst = minibatch(xtst, ytst, batchsize; xtype=atype)
    report(epoch) = begin
            bw = binarize(w)
            acctrn = accuracy(bw, dtrn, (w,x) ->predict(w,x,bmom))
            acctst = accuracy(bw, dtst, (w,x) ->predict(w,x,bmom))
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


#    report(0); 
tic()
    @time for epoch=1:epochs
        for (x, y) in  minibatch(xtrn, ytrn, batchsize, shuffle=true, xtype=atype)
            bw = binarize(w)
            dw = grad(loss)(bw, x, y, bmom; pdrop=pdrop, input_do = input_do)
            update!(w, dw, opt)
            for i=1:3:length(w)
              w[i] = clip(w[i], 1e-2)
            end
        end
        (epoch % infotime == 0) && (report(epoch); toc(); tic())
        # acctrn == 200 && break
    end; toq()
    println("# FINAL RESULT acccuracy: train=$acctrn test=$acctst" , )
    return w, acctst
end
